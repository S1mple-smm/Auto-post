#!/usr/bin/env python3
"""
Telegram ZIP Bot
- You send a ZIP file in private chat
- Bot auto-groups photos by product using Gemini Vision
- Posts each product as a separate album to your channel
- Reports progress back to you
"""

import os, sys, json, asyncio, zipfile, shutil, io, time, logging, re
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Deps ───────────────────────────────────────────────────────────────────────
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                              ContextTypes, filters)
    from telegram.constants import ParseMode
except ImportError:
    sys.exit("❌  Run:  pip install python-telegram-bot")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("❌  Run:  pip install google-genai")

try:
    from PIL import Image as PILImage
except ImportError:
    sys.exit("❌  Run:  pip install Pillow")

# ── Config (from environment variables) ───────────────────────────────────────
BOT_TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]    # bot that receives ZIPs
CHANNEL_ID     = os.environ["TELEGRAM_CHAT_ID"]      # channel to post products
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]
ALLOWED_USERS  = set(os.getenv("ALLOWED_USER_IDS", "").split(","))  # comma-separated user IDs

SHOP_CONFIG_FILE = "shop.json"
SUPPORTED_EXT    = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ALBUM_SIZE   = 10
GEMINI_RPM       = 15
GEMINI_RPD       = 500

# ── Rate limiter ───────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm, rpd):
        self.rpm = rpm; self.rpd = rpd
        self.minute_log = []; self.day_log = []

    def wait_if_needed(self):
        now = time.time()
        self.minute_log = [t for t in self.minute_log if now - t < 60]
        self.day_log    = [t for t in self.day_log    if now - t < 86400]

        if len(self.day_log) >= self.rpd:
            wait = 86400 - (now - self.day_log[0])
            raise RuntimeError(f"Daily Gemini limit ({self.rpd}/day) reached. Try in {int(wait//3600)}h {int((wait%3600)//60)}m.")

        if len(self.minute_log) >= self.rpm:
            wait = 60 - (now - self.minute_log[0]) + 1
            log.info(f"Rate limit: waiting {int(wait)}s …")
            time.sleep(wait)
            now = time.time()
            self.minute_log = [t for t in self.minute_log if now - t < 60]

        self.minute_log.append(time.time())
        self.day_log.append(time.time())

    @property
    def remaining_today(self):
        now = time.time()
        self.day_log = [t for t in self.day_log if now - t < 86400]
        return self.rpd - len(self.day_log)

rate_limiter = RateLimiter(GEMINI_RPM, GEMINI_RPD)

# ── Shop config ────────────────────────────────────────────────────────────────
def load_shop_config() -> dict:
    if os.path.exists(SHOP_CONFIG_FILE):
        with open(SHOP_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

# ── Gemini helpers ─────────────────────────────────────────────────────────────
def load_image_bytes(path: str) -> bytes:
    with PILImage.open(path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()

def gemini_call(parts: list, max_retries: int = 5) -> str:
    rate_limiter.wait_if_needed()
    client = genai.Client(api_key=GEMINI_KEY)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite", contents=parts)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if ("503" in err or "429" in err) and attempt < max_retries:
                wait = attempt * 15
                log.info(f"Gemini busy, retrying in {wait}s (attempt {attempt}) …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted Gemini retries")

def describe_single_image(path: str, index: int) -> dict:
    """
    Ask Gemini to describe ONE image — optimized to handle mixed collages 
    and prioritize real photos to be the primary cover slot.
    """
    with PILImage.open(path) as img:
        img.thumbnail((500, 500))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=75)
        img_bytes = buf.getvalue()

    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text="""
Analyze this sports kit item image. Note that this file might be a combination multi-window collage.

Identify the following details from the dominant view:
1. "primary_color" — The main background fabric color (e.g., "blue", "white", "red", "yellow", "green").
2. "secondary_color" — Accent or collar/sleeve trim colors (e.g., "gold", "navy", "black").
3. "pattern" — Brief note of design traits (e.g., "zigzag", "stripes", "plain").
4. "garment" — Usually "jersey+shorts" or "jersey".
5. "type" — Choose "render" ONLY if the asset is an absolute digital 3D vector model graphic on a plain background. Choose "photo" if it is a real physical fabric sample flat-laid on a floor, hung up, or showing organic wrinkles/shadows.
6. "view" — Label the perspective:
   - "front": if the chest and main crest are centered.
   - "back": if highlighting player numbers or rear face.
   - "collage": if it contains both perspectives side-by-side or combined.

Return ONLY a clean JSON schema:
{"primary_color": "...", "secondary_color": "...", "pattern": "...", "garment": "...", "type": "render", "view": "front"}
""")
    ]
    raw = gemini_call(parts)
    raw = raw.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(match.group()) if match else {
            "primary_color": f"unknown_{index}", "secondary_color": "", "pattern": "", "garment": "", "type": "photo", "view": "front"
        }
    data["_index"] = index
    data["_path"]  = path
    return data


def match_fingerprints(descriptions: list) -> list:
    """
    Step 2: send all the lightweight text fingerprints (no images) to Gemini in ONE
    cheap text-only call, and ask it to cluster which indices belong to the same product.
    This catches cases where front/back/render/photo of the same kit got slightly
    different color/pattern wording in step 1.
    """
    fingerprint_lines = []
    for d in descriptions:
        fingerprint_lines.append(
            f"{d['_index']}: primary={d.get('primary_color')}, secondary={d.get('secondary_color')}, "
            f"pattern={d.get('pattern')}, garment={d.get('garment')}, type={d.get('type')}"
        )
    listing = "\n".join(fingerprint_lines)

    prompt = f"""Below is a list of product image fingerprints, one per line, format:
INDEX: primary=color, secondary=color, pattern=desc, garment=type, type=render/photo

{listing}

Cluster these indices into groups — each group = images of the SAME physical product
(same color combo + same pattern), regardless of whether they are front/back/render/photo.
Minor wording differences in pattern description can still mean the same product if colors match closely.
Different primary OR secondary color = different product.

Reply ONLY with a JSON array of groups of indices, e.g. [[0,1,2],[3,4]]
No explanation, no markdown — ONLY the JSON array.
"""
    raw = gemini_call([types.Part.from_text(text=prompt)])
    raw = raw.strip().strip("```json").strip("```").strip()
    try:
        groups_idx = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[\s*\[.*?\]\s*\]', raw, re.DOTALL)
        groups_idx = json.loads(match.group()) if match else None

    if not groups_idx:
        # Fallback: each image its own group
        return [[d["_index"]] for d in descriptions]
    return groups_idx

def ai_group_images(image_paths: list) -> list:
    """
    Step 1: describe each image's visual fingerprint individually (reliable for lite models).
    Step 2: cluster fingerprints into product groups via one cheap text-only call.
    Step 3: sort each group renders-first, photos-last.
    """
    descriptions = []
    for i, path in enumerate(image_paths):
        try:
            d = describe_single_image(path, i)
        except Exception as e:
            log.warning(f"Failed to describe image {i}: {e}")
            d = {"primary_color": f"unknown_{i}", "secondary_color": "", "pattern": "",
                 "garment": "", "type": "photo", "_index": i, "_path": path}
        descriptions.append(d)
        log.info(f"  IMAGE_{i}: {d.get('primary_color')}/{d.get('secondary_color')} "
                 f"pattern='{d.get('pattern')}' type='{d.get('type')}'")

        by_index = {d["_index"]: d for d in descriptions}

    try:
        groups_idx = match_fingerprints(descriptions)
    except Exception as e:
        log.warning(f"Fingerprint matching failed, falling back to 1 group per image: {e}")
        groups_idx = [[d["_index"]] for d in descriptions]

    # Build final groups with inverted priority sorting for Telegram's Grid Engine
    result = []
    for group in groups_idx:
        items = [by_index[i] for i in group if i in by_index]
        if not items:
            continue
        
        def sorting_key(item):
            g_type = str(item.get("type", "photo")).lower().strip()
            g_view = str(item.get("view", "front")).lower().strip()
            
            # INVERTED SCALE: Lowest weight values are placed first (Index 0)
            # This forces the physical product photo to fill Telegram's large layout slot.
            if g_type == "photo":
                if g_view == "front":    return 0  # 1st: Real Photo Front (Becomes large cover)
                if g_view == "collage":  return 1  # 2nd: Real Photo Mixed Composition
                if g_view == "back":     return 2  # 3rd: Real Photo Back Area
                return 3                           # 4th: Real Photo Miscellaneous
            else:
                if g_view == "front":    return 4  # 5th: Render Mockup Front
                if g_view == "back":     return 5  # 6th: Render Mockup Back
                if g_view == "collage":  return 6  # 7th: Render Mixed Collage Split
                return 7                           # 8th: Render Miscellaneous

        # Sort this kit's images according to the array priority metric
        items_sorted = sorted(items, key=sorting_key)
        
        # Save the optimized file paths sequence
        result.append([d["_path"] for d in items_sorted])

    return result or [image_paths]

def build_caption_prompt(cfg: dict) -> str:
    lang_map  = {"ru": "Russian", "uz": "Uzbek", "en": "English"}
    lang      = lang_map.get(cfg.get("language", "ru"), "Russian")
    uzum_line = f"Наш магазин на Узум:\n{cfg['uzum_link']}\n" if cfg.get("uzum_link") else ""
    tg_line   = f"Telegram: {cfg['telegram_link']}\n"          if cfg.get("telegram_link") else ""
    admin     = cfg.get("admin_tag", "")
    price     = cfg.get("price", "")
    delivery  = cfg.get("delivery", "")
    sizes     = cfg.get("sizes", "S, M, L, XL, 2XL")

    example = (
        f"Футбольная форма Реал Мадрид:\n"
        f"⚽Многофункциональная: Идеально подходит для футбола, бега и других видов спорта.\n"
        f"📐Размеры: Доступны размеры {sizes}.\n"
        f"🧵 Материал: Легкий и дышащий полиэстер.\n"
        f"🎨 Стиль: Спортивный, с коротким рукавом.\n"
        f"👕Комфорт: Свободная посадка для удобства в движении.\n"
        f"🚚Доставка осуществляется в течении {delivery}\n"
        f"💰{price}\n\n"
        f"{uzum_line}{tg_line}{admin}"
    )
    return (
        f"You are a product copywriter for a sportswear shop.\n"
        f"Look at the product image(s) and write a Telegram post in {lang} "
        f"EXACTLY following this style:\n\n{example}\n\nRules:\n"
        f"- First line: product name + colon (no emoji on first line)\n"
        f"- Emoji bullets for each feature\n"
        f"- Include: purpose, sizes ({sizes}), material, style, comfort, delivery ({delivery}), price ({price})\n"
        f"- End with shop links and admin tag exactly as shown\n"
        f"- Output ONLY the post text, no markdown, no backticks"
    )

def generate_description(image_paths: list, cfg: dict) -> str:
    parts = []
    for path in image_paths[:4]:
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=build_caption_prompt(cfg)))
    return gemini_call(parts)

# ── Telegram posting ───────────────────────────────────────────────────────────
async def post_album_to_channel(bot, image_paths: list, caption: str) -> bool:
    from telegram import InputMediaPhoto
    
    # 💡 TELEGRAM GRID HACK:
    # Telegram only builds the "1 Large Cover + 2 Small Sidekas" layout if there are EXACTLY 3 images.
    # If there are 4 images, it forces a flat 2x2 square grid. 
    # Let's slice the array to a maximum of 3 to force the layout you want.
    layout_paths = image_paths[:3] 
    
    media_group = []
    open_files = []
    
    try:
        for i, path in enumerate(layout_paths):
            f = open(path, "rb")
            open_files.append(f)
            
            if i == 0:
                # The first item in the array becomes the dominant large image and holds the caption
                media_group.append(
                    InputMediaPhoto(media=f, caption=caption, parse_mode="HTML")
                )
            else:
                media_group.append(InputMediaPhoto(media=f))
        
        # Send the group using the native library
        await bot.send_media_group(chat_id=CHANNEL_ID, media=media_group)
        return True
        
    except Exception as e:
        log.error(f"Failed to post album: {e}")
        return False
        
    finally:
        # Always close files to prevent memory leaks
        for f in open_files:
            f.close()


# ── Extract images from ZIP ────────────────────────────────────────────────────
def extract_all_images(zip_path: str, dest_dir: str) -> list:
    paths = []
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(sorted(zf.namelist())):
            if "__MACOSX" in name or name.startswith(".") or "/." in name:
                continue
            p = Path(name)
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            out = os.path.join(dest_dir, f"{i:04d}_{p.name}")
            with zf.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
            paths.append(out)
    return sorted(paths)

# ── Access control ─────────────────────────────────────────────────────────────
def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS or ALLOWED_USERS == {''}:
        return True   # no restriction set
    return str(user_id) in ALLOWED_USERS

# ── Bot handlers ───────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    cfg = load_shop_config()
    shop = cfg.get("shop_name", "your shop") if cfg else "not configured yet"
    remaining = rate_limiter.remaining_today

    await update.message.reply_text(
        f"👋 *Smart Product Poster*\n\n"
        f"🏪 Shop: {shop}\n"
        f"📊 Gemini quota today: {remaining}/{GEMINI_RPD} requests left\n\n"
        f"Just send me a *ZIP file* with product photos — I'll auto-group them by product "
        f"and post each one to your channel with an AI-generated description.\n\n"
        f"Commands:\n"
        f"/start — show this message\n"
        f"/quota — check remaining Gemini quota\n"
        f"/shop — show current shop config",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_quota(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    r = rate_limiter.remaining_today
    await update.message.reply_text(
        f"📊 Gemini quota: *{r}/{GEMINI_RPD}* requests remaining today.",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_shop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    cfg = load_shop_config()
    if not cfg:
        await update.message.reply_text("⚠️ No shop config found. Edit shop.json on the server.")
        return
    text = "\n".join(f"*{k}:* {v}" for k, v in cfg.items())
    await update.message.reply_text(f"🏪 *Shop config:*\n\n{text}", parse_mode=ParseMode.MARKDOWN)

async def handle_zip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Main handler: receives ZIP, processes, posts to channel."""
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ You are not authorized to use this bot.")
        return

    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".zip"):
        await update.message.reply_text("📎 Please send a ZIP file.")
        return

    cfg = load_shop_config()
    if not cfg:
        await update.message.reply_text(
            "⚠️ Shop config not found!\n"
            "Create shop.json on the server first (see README)."
        )
        return

    remaining = rate_limiter.remaining_today
    status_msg = await update.message.reply_text(
        f"📦 Received *{doc.file_name}*\n"
        f"📊 Gemini quota: {remaining}/{GEMINI_RPD} left today\n\n"
        f"⏳ Downloading…",
        parse_mode=ParseMode.MARKDOWN
    )

    tmpdir = f"/tmp/poster_{update.message.message_id}"
    os.makedirs(tmpdir, exist_ok=True)

    try:
        # Download ZIP
        zip_path = os.path.join(tmpdir, doc.file_name)
        tg_file  = await ctx.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(zip_path)

        await status_msg.edit_text(
            f"📦 *{doc.file_name}* downloaded\n"
            f"🤖 Analysing each photo individually (team/color/type)…",
            parse_mode=ParseMode.MARKDOWN
        )

        # Extract & group
        all_images = extract_all_images(zip_path, tmpdir)
        if not all_images:
            await status_msg.edit_text("❌ No images found in the ZIP file.")
            return

        # AI grouping — 1 Gemini request PER IMAGE (more reliable with lite model)
        loop = asyncio.get_event_loop()
        groups = await loop.run_in_executor(None, ai_group_images, all_images)

        total = len(groups)
        await status_msg.edit_text(
            f"📦 *{doc.file_name}*\n"
            f"🖼 Found *{total} product(s)* — starting to post…\n"
            f"📊 Quota: {rate_limiter.remaining_today}/{GEMINI_RPD} left",
            parse_mode=ParseMode.MARKDOWN
        )

        # Post each group
        posted = 0
        failed = 0
        for i, images in enumerate(groups, 1):
            try:
                await status_msg.edit_text(
                    f"📦 *{doc.file_name}*\n"
                    f"⏳ Processing product *{i}/{total}* ({len(images)} photos)…\n"
                    f"📊 Quota: {rate_limiter.remaining_today}/{GEMINI_RPD} left",
                    parse_mode=ParseMode.MARKDOWN
                )

                # Generate description in thread pool (blocking)
                caption = await loop.run_in_executor(
                    None, generate_description, images, cfg)

                # Post album to channel
                for chunk_start in range(0, len(images), MAX_ALBUM_SIZE):
                    chunk = images[chunk_start:chunk_start + MAX_ALBUM_SIZE]
                    ok = await post_album_to_channel(ctx.bot, chunk,
                                                     caption if chunk_start == 0 else "")
                    if ok:
                        posted += 1
                    else:
                        failed += 1
                    await asyncio.sleep(2)

                # Wait between posts (rate limit buffer)
                if i < total:
                    await asyncio.sleep(12)

            except RuntimeError as e:
                if "Daily Gemini limit" in str(e):
                    await status_msg.edit_text(
                        f"⛔ *Daily quota exhausted!*\n\n"
                        f"Posted {posted}/{total} products before hitting the limit.\n"
                        f"{str(e)}",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return
                else:
                    log.error(f"Product {i} failed: {e}")
                    failed += 1
                    continue

        # Final report
        icon = "✅" if failed == 0 else "⚠️"
        await status_msg.edit_text(
            f"{icon} *Done!*\n\n"
            f"📮 Posted: *{posted}* product(s) to channel\n"
            f"❌ Failed: *{failed}*\n"
            f"📊 Gemini quota remaining: *{rate_limiter.remaining_today}/{GEMINI_RPD}*",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        log.exception("Unexpected error")
        await status_msg.edit_text(f"❌ Unexpected error:\n<code>{e}</code>",
                                   parse_mode=ParseMode.HTML)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting Smart Product Poster bot…")

    if not load_shop_config():
        log.warning("⚠️  shop.json not found — bot will warn users until it's created.")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("quota", cmd_quota))
    app.add_handler(CommandHandler("shop",  cmd_shop))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_zip))

    log.info("Bot is running. Send a ZIP file to start.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

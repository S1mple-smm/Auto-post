#!/usr/bin/env python3
"""
Telegram ZIP Bot
- You send a ZIP file in private chat
- Bot auto-groups photos by product using Gemini Vision
- Posts each product as a separate album to your channel
- Applies custom user-selected description templates
"""

import os, sys, json, asyncio, zipfile, shutil, io, time, logging, re
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from telethon import TelegramClient

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Deps ───────────────────────────────────────────────────────────────────────
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                              CallbackQueryHandler, filters)
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
API_ID         = int(os.environ["TELEGRAM_API_ID"])      # From my.telegram.org
API_HASH       = os.environ["TELEGRAM_API_HASH"]        # From my.telegram.org
BOT_TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]    
CHANNEL_ID     = os.environ["TELEGRAM_CHAT_ID"]      
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]
ALLOWED_USERS  = set(os.getenv("ALLOWED_USER_IDS", "").split(","))

SHOP_CONFIG_FILE = "shop.json"
SUPPORTED_EXT    = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ALBUM_SIZE   = 10
GEMINI_RPM       = 15
GEMINI_RPD       = 500

# Initialize Telethon client (MTProto for >20MB downloads)
telethon_client = TelegramClient("bot_session", API_ID, API_HASH)

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

# ── Config Helpers ──────────────────────────────────────────────────────────────
def escape_md(text: str) -> str:
    if not text: return text
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text

def load_shop_config() -> dict:
    if os.path.exists(SHOP_CONFIG_FILE):
        with open(SHOP_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

# ── Description templates ───────────────────────────────────────────────────────
TEMPLATES_FILE = "templates.json"

def load_templates() -> dict:
    if os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_templates(templates: dict):
    with open(TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

# ── Gemini helpers ─────────────────────────────────────────────────────────────
def load_image_bytes(path: str, max_size=(800, 800)) -> bytes:
    with PILImage.open(path) as img:
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()

def gemini_call(parts: list, max_retries: int = 5) -> str:
    rate_limiter.wait_if_needed()
    client = genai.Client(api_key=GEMINI_KEY)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=parts)
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

def ai_group_images(image_paths: list) -> list:
    """Passes all images to Gemini for structural clustering with strict position and color rules."""
    parts = []
    for i, path in enumerate(image_paths):
        parts.append(types.Part.from_text(text=f"Image Index: {i}"))
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
        
    prompt = """
    You are an expert sports apparel merchandiser. Look at all the provided images.
    Group the images that show the EXACT same physical product.

    CRITICAL GROUPING RULES (DO NOT FAIL THESE):
    1. STRICT COLOR MATCHING: If Image A is RED and Image B is WHITE, they are DIFFERENT products. NEVER group different colors together, even if they are the exact same brand or design template. Color is the #1 deciding factor.
    2. FRONT & BACK PAIRING: A back view of a shirt belongs ONLY with the front view of the EXACT SAME COLOR and pattern. Do not mix colors!

    CRITICAL SORTING RULES (WITHIN EACH GROUP):
    You must distinguish between "3D Renders" (pure white background, invisible ghost mannequin) and "Real Photos" (flat-lays on grey floor, wrinkles, visible shorts next to shirt).
    
    For each valid product group, identify the image indices for:
    - "front_view": MUST be the Front-facing 3D Render (white background). IF AND ONLY IF no 3D render exists for this specific colorway, use the best front-facing real photo.
    - "back_view": MUST be the Back-facing 3D Render. IF AND ONLY IF no back 3D render exists, use the back real photo. (Use null if there is no back view).
    - "other_photos": Put ALL remaining photos (like the grey flat-lays if renders were used, close-ups, shorts alone) in this array.

    EVERY SINGLE IMAGE INDEX MUST APPEAR EXACTLY ONCE somewhere in your output.

    Return ONLY a valid JSON array of objects. Example:
    [
      {
        "front_view": 2,
        "back_view": 5,
        "other_photos": [0, 1]
      }
    ]
    Do not include any markdown, backticks, or explanations. Just the JSON array.
    """
    parts.append(types.Part.from_text(text=prompt))
    
    log.info(f"Sending {len(image_paths)} images to Gemini for structural clustering...")
    
    try:
        raw = gemini_call(parts)
        raw = raw.strip().strip("```json").strip("```").strip()
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            structured_groups = json.loads(match.group())
        else:
            structured_groups = json.loads(raw)
    except Exception as e:
        log.warning(f"Direct clustering failed ({e}), falling back to 1 group per image.")
        return [[p] for p in image_paths]
        
    result = []
    seen_indices = set()
    for group in structured_groups:
        current_group_paths = []
        
        front_idx = group.get("front_view")
        if isinstance(front_idx, int) and 0 <= front_idx < len(image_paths) and front_idx not in seen_indices:
            current_group_paths.append(image_paths[front_idx])
            seen_indices.add(front_idx)
            
        back_idx = group.get("back_view")
        if isinstance(back_idx, int) and 0 <= back_idx < len(image_paths) and back_idx not in seen_indices:
            current_group_paths.append(image_paths[back_idx])
            seen_indices.add(back_idx)
            
        for other_idx in group.get("other_photos", []) or []:
            if isinstance(other_idx, int) and 0 <= other_idx < len(image_paths) and other_idx not in seen_indices:
                current_group_paths.append(image_paths[other_idx])
                seen_indices.add(other_idx)
                
        if current_group_paths:
            result.append(current_group_paths)

    missing = [i for i in range(len(image_paths)) if i not in seen_indices]
    if missing:
        for idx in missing:
            result.append([image_paths[idx]])

    return result or [image_paths]

def detect_product_category(image_paths: list) -> str:
    parts = []
    for path in image_paths[:3]:
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text="""
Reply with ONLY one word: "jersey" if it's a shirt/kit, "boots" if it's footwear, or "other" if accessory.
"""))
    try:
        raw = gemini_call(parts).strip().lower()
        raw = re.sub(r'[^a-z]', '', raw)
        if raw in ("jersey", "boots", "other"):
            return raw
    except Exception as e:
        log.warning(f"Category detection failed: {e}")
    return "jersey"

CATEGORY_PROFILES = {
    "jersey": {
        "sizes_key": "sizes",
        "default_sizes": "S, M, L, XL, 2XL",
        "example": (
            "Футбольная форма клуба/сборной:\n"
            "⚽Многофункциональная: Идеально подходит для футбола и бега.\n"
            "📐Размеры: В наличии размеры {sizes}.\n"
            "🧵Материал: Легкий и дышащий полиэстер\n"
            "👕Комфорт: Свободная и удобная посадка.\n"
            "🚚Доставка осуществляется в течении {delivery}\n"
            "💰{price}\n\n"
            "{uzum_line}{tg_line}{admin}"
        ),
    },
    "boots": {
        "sizes_key": "shoe_sizes",
        "default_sizes": "38, 39, 40, 41, 42, 43, 44, 45",
        "example": (
            "Футбольные бутсы:\n"
            "⚽Назначение: Для игры на натуральном и искусственном газоне.\n"
            "📐Размеры: В наличии размеры {sizes}.\n"
            "👟Комфорт: Плотная посадка для уверенного контроля.\n"
            "🚚Доставка осуществляется в течении {delivery}\n"
            "💰{price}\n\n"
            "{uzum_line}{tg_line}{admin}"
        ),
    },
    "other": {
        "sizes_key": "sizes",
        "default_sizes": "Универсальный",
        "example": (
            "Спортивный аксессуар:\n"
            "⚽Многофункциональный: Идеально подходит для тренировок.\n"
            "📐Размеры: {sizes}.\n"
            "🚚Доставка осуществляется в течении {delivery}\n"
            "💰{price}\n\n"
            "{uzum_line}{tg_line}{admin}"
        ),
    },
}

def build_caption_prompt(cfg: dict, category: str, tmpl_text: str = "") -> str:
    lang_map  = {"ru": "Russian", "uz": "Uzbek", "en": "English"}
    lang      = lang_map.get(cfg.get("language", "ru"), "Russian")
    uzum_line = f"Наш магазин на Узум:\n{cfg['uzum_link']}\n" if cfg.get("uzum_link") else ""
    tg_line   = f"Telegram: {cfg['telegram_link']}\n"          if cfg.get("telegram_link") else ""
    admin     = cfg.get("admin_tag", "")
    price     = cfg.get("price", "")
    delivery  = cfg.get("delivery", "")

    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["jersey"])
    sizes   = cfg.get(profile["sizes_key"]) or cfg.get("sizes") or profile["default_sizes"]

    if tmpl_text:
        return (
            f"You are a product copywriter for a sportswear shop.\n"
            f"Look at the product image(s) and write a Telegram post in {lang}.\n\n"
            f"Here is the CUSTOM TEMPLATE you must follow strictly:\n"
            f"---------------------\n"
            f"{tmpl_text}\n"
            f"---------------------\n\n"
            f"CRITICAL RULES:\n"
            f"1. Use the EXACT structure, emojis, and text layout from the custom template.\n"
            f"2. Adapt the product name on the very first line to match the visual features of the image (e.g. model name and color).\n"
            f"3. Ensure shop details are injected if missing (Sizes: {sizes}, Price: {price}, Delivery: {delivery}).\n"
            f"4. Always append the shop links at the very bottom:\n{uzum_line}{tg_line}{admin}\n"
            f"5. DO NOT use brand names (like Adidas, Nike).\n"
            f"6. DO NOT write two separate descriptions. Output ONLY the final completed template."
        )
    else:
        example = profile["example"].format(
            sizes=sizes, delivery=delivery, price=price,
            uzum_line=uzum_line, tg_line=tg_line, admin=admin
        )
        return (
            f"You are a product copywriter for a sportswear shop.\n"
            f"Look at the product image(s) and write a Telegram post in {lang} "
            f"EXACTLY following this style:\n\n{example}\n\nRules:\n"
            f"- First line: ONLY the model name + color. DO NOT use brand names in the title or description.\n"
            f"- Emoji bullets for each feature\n"
            f"- Use wording appropriate for this exact product type.\n"
            f"- Output ONLY the post text, no markdown, no backticks"
        )

def generate_description(image_paths: list, cfg: dict, tmpl_text: str = "") -> str:
    category = detect_product_category(image_paths)
    log.info(f"  Detected category: {category}")
    parts = []
    for path in image_paths[:4]:
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
        
    parts.append(types.Part.from_text(text=build_caption_prompt(cfg, category, tmpl_text)))
    return gemini_call(parts)

# ── Telegram posting ───────────────────────────────────────────────────────────
async def post_album_to_channel(bot, image_paths: list, caption: str):
    import aiohttp as _aio
    import html as _html
    base = f"https://api.telegram.org/bot{BOT_TOKEN}"
    safe_caption = _html.escape(caption, quote=False) if caption else caption

    if len(image_paths) == 1:
        path = image_paths[0]
        try:
            with open(path, "rb") as f:
                async with _aio.ClientSession() as session:
                    form = _aio.FormData()
                    form.add_field("chat_id", str(CHANNEL_ID))
                    if safe_caption:
                        form.add_field("caption", safe_caption)
                        form.add_field("parse_mode", "HTML")
                    form.add_field("photo", f, filename=Path(path).name, content_type="image/jpeg")
                    async with session.post(f"{base}/sendPhoto", data=form) as resp:
                        result = await resp.json()
                        if result.get("ok"): return True, ""
                        return False, result.get("description", "Telegram API error")
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    media = []
    files = {}

    for i, path in enumerate(image_paths):
        field = f"photo{i}"
        files[field] = open(path, "rb")
        entry = {"type": "photo", "media": f"attach://{field}"}
        if i == 0:
            entry["caption"]    = safe_caption
            entry["parse_mode"] = "HTML"
        media.append(entry)

    try:
        async with _aio.ClientSession() as session:
            form = _aio.FormData()
            form.add_field("chat_id", str(CHANNEL_ID))
            form.add_field("media",   json.dumps(media))
            for field, fobj in files.items():
                form.add_field(field, fobj, filename=Path(fobj.name).name, content_type="image/jpeg")
            async with session.post(f"{base}/sendMediaGroup", data=form) as resp:
                result = await resp.json()
                if result.get("ok"): return True, ""
                return False, result.get("description", "Telegram API error")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        for fobj in files.values():
            fobj.close()

def extract_all_images(zip_path: str, dest_dir: str) -> list:
    paths = []
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(sorted(zf.namelist())):
            if "__MACOSX" in name or name.startswith(".") or "/." in name: continue
            p = Path(name)
            if p.suffix.lower() not in SUPPORTED_EXT: continue
            out = os.path.join(dest_dir, f"{i:04d}_{p.name}")
            with zf.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
            paths.append(out)
    return sorted(paths)

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS or ALLOWED_USERS == {''}: return True
    return str(user_id) in ALLOWED_USERS

# ── Bot handlers ───────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    cfg = load_shop_config()
    shop = cfg.get("shop_name", "your shop") if cfg else "not configured yet"
    await update.message.reply_text(
        f"👋 *Smart Product Poster*\n\n🏪 Shop: {shop}\n"
        f"Send me a *ZIP file* with product photos. I'll auto-group them, "
        f"write descriptions, and post them as albums.\n\n"
        f"/newtemplate Name | text — save a reusable text block\n"
        f"/templates — list saved templates", parse_mode=ParseMode.MARKDOWN)

async def cmd_quota(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    await update.message.reply_text(f"📊 Gemini quota: *{rate_limiter.remaining_today}/{GEMINI_RPD}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_shop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    cfg = load_shop_config()
    text = "\n".join(f"*{k}:* {v}" for k, v in cfg.items()) if cfg else "⚠️ No config"
    await update.message.reply_text(f"🏪 *Shop config:*\n\n{text}", parse_mode=ParseMode.MARKDOWN)

async def cmd_newtemplate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    raw = update.message.text.partition(" ")[2].strip()
    if "|" not in raw:
        await update.message.reply_text("📝 *Usage:*\n`/newtemplate Name | text`", parse_mode=ParseMode.MARKDOWN)
        return
    name, _, body = raw.partition("|")
    name, body = name.strip(), body.strip()
    templates = load_templates()
    templates[name] = body
    save_templates(templates)
    await update.message.reply_text(f"✅ Template saved as *{escape_md(name)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_templates(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    templates = load_templates()
    lines = [f"• *{escape_md(name)}*\n  {escape_md(body[:80])}…" for name, body in templates.items()]
    await update.message.reply_text("📋 *Saved templates:*\n\n" + "\n\n".join(lines) if lines else "📭 None saved.", parse_mode=ParseMode.MARKDOWN)

async def cmd_deltemplate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    name = update.message.text.partition(" ")[2].strip()
    templates = load_templates()
    if name in templates:
        del templates[name]
        save_templates(templates)
        await update.message.reply_text(f"🗑 Deleted *{escape_md(name)}*", parse_mode=ParseMode.MARKDOWN)

async def handle_zip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".zip"): return

    cfg = load_shop_config()
    status_msg = await update.message.reply_text(f"📦 Downloading via MTProto…", parse_mode=ParseMode.MARKDOWN)

    tmpdir = f"/tmp/poster_{update.message.message_id}"
    os.makedirs(tmpdir, exist_ok=True)
    zip_path = os.path.join(tmpdir, doc.file_name)

    try:
        # Download via Telethon client bypassing Bot API limit
        await telethon_client.start(bot_token=BOT_TOKEN)
        telethon_msg = await telethon_client.get_messages(update.effective_chat.id, ids=update.message.message_id)
        await telethon_client.download_media(telethon_msg.media, file=zip_path)

        await status_msg.edit_text(f"🤖 Analyzing photos and grouping...", parse_mode=ParseMode.MARKDOWN)

        all_images = extract_all_images(zip_path, tmpdir)
        loop = asyncio.get_event_loop()
        groups = await loop.run_in_executor(None, ai_group_images, all_images)

        total = len(groups)
        templates = load_templates()

        batch_key = f"{update.effective_chat.id}_{update.message.message_id}"
        buttons = [[InlineKeyboardButton("🚫 No template — post as is", callback_data=f"batchtmpl::__none__::{batch_key}")]]
        for name in templates.keys():
            buttons.append([InlineKeyboardButton(f"📋 {name}", callback_data=f"batchtmpl::{name}::{batch_key}")])
        keyboard = InlineKeyboardMarkup(buttons)

        event = asyncio.Event()
        ctx.bot_data.setdefault("pending_batches", {})[batch_key] = {"event": event, "template": None}

        await status_msg.edit_text(
            f"📦 *{escape_md(doc.file_name)}*\n🖼 Found *{total} product(s)*\n\nChoose a template style:",
            parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)

        await event.wait()
        chosen = ctx.bot_data["pending_batches"].pop(batch_key, {})
        tmpl_name = chosen.get("template") or "__none__"
        tmpl_text = templates.get(tmpl_name, "") if tmpl_name != "__none__" else ""

        posted, failed, error_details = 0, 0, []

        for i, images in enumerate(groups, 1):
            try:
                await status_msg.edit_text(f"⏳ Processing product *{i}/{total}*…", parse_mode=ParseMode.MARKDOWN)
                
                caption = await loop.run_in_executor(None, generate_description, images, cfg, tmpl_text)

                ok_all, post_errors = True, []
                for chunk_start in range(0, len(images), MAX_ALBUM_SIZE):
                    chunk = images[chunk_start:chunk_start + MAX_ALBUM_SIZE]
                    ok, err_msg = await post_album_to_channel(ctx.bot, chunk, caption if chunk_start == 0 else "")
                    ok_all = ok_all and ok
                    if not ok: post_errors.append(err_msg)
                    await asyncio.sleep(2)

                if ok_all:
                    posted += 1
                else:
                    failed += 1
                    error_details.append(f"Product {i}: " + "; ".join(post_errors))

                if i < total: await asyncio.sleep(12)
            except Exception as e:
                failed += 1
                log.exception(f"Unexpected error on product {i}: {e}")

        report = f"✅ *Done!*\n\n📮 Posted: *{posted}*\n❌ Failed: *{failed}*"
        await status_msg.edit_text(report, parse_mode=ParseMode.MARKDOWN)
        if error_details: await update.message.reply_text("🔍 Errors:\n" + "\n".join(error_details))

    except Exception as e:
        await status_msg.edit_text(f"❌ Error:\n<code>{e}</code>", parse_mode=ParseMode.HTML)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

async def handle_template_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not is_allowed(query.from_user.id): return
    try:
        _, tmpl_name, batch_key = query.data.split("::", 2)
    except ValueError: return

    pending = ctx.bot_data.get("pending_batches", {}).get(batch_key)
    if not pending: return
    pending["template"] = tmpl_name
    pending.get("event").set()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("quota", cmd_quota))
    app.add_handler(CommandHandler("shop",  cmd_shop))
    app.add_handler(CommandHandler("newtemplate", cmd_newtemplate))
    app.add_handler(CommandHandler("templates",   cmd_templates))
    app.add_handler(CommandHandler("deltemplate", cmd_deltemplate))
    app.add_handler(CallbackQueryHandler(handle_template_choice, pattern=r"^batchtmpl::"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_zip))
    log.info("Bot is running. Send a ZIP file to start.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
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
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from telethon import TelegramClient

# ── New MTProto Configuration ────────────────────────────────────────────────
API_ID   = int(os.environ["TELEGRAM_API_ID"])     # From my.telegram.org
API_HASH = os.environ["TELEGRAM_API_HASH"]       # From my.telegram.org
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Initialize Telethon client in Bot Login Mode
# This allows Telethon to use your bot's identity to download files natively
telethon_client = TelegramClient("bot_session", API_ID, API_HASH)

# ── Updated ZIP Handler ──────────────────────────────────────────────────────
async def handle_zip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Main handler: receives ZIP, downloads via MTProto, processes, posts."""
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ You are not authorized to use this bot.")
        return

    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".zip"):
        await update.message.reply_text("📎 Please send a ZIP file.")
        return

    cfg = load_shop_config()
    if not cfg:
        await update.message.reply_text("⚠️ Shop config not found! Create shop.json on the server.")
        return

    remaining = rate_limiter.remaining_today
    status_msg = await update.message.reply_text(
        f"📦 Received *{doc.file_name}*\n"
        f"📊 Gemini quota: {remaining}/{GEMINI_RPD} left today\n\n"
        f"⏳ Downloading large file directly via MTProto…",
        parse_mode=ParseMode.MARKDOWN
    )

    tmpdir = f"/tmp/poster_{update.message.message_id}"
    os.makedirs(tmpdir, exist_ok=True)
    zip_path = os.path.join(tmpdir, doc.file_name)

    try:
        # === THE NEW TELETHON DOWNLOADER ===
        # This completely ignores the 20MB Bot API limit
        async with telethon_client:
            telethon_msg = await telethon_client.get_messages(
                update.effective_chat.id, 
                ids=update.message.message_id
            )
            await telethon_client.download_media(telethon_msg.media, file=zip_path)

        await status_msg.edit_text(
            f"📦 *{doc.file_name}* downloaded successfully!\n"
            f"🤖 Analyzing photos and sorting into albums…",
            parse_mode=ParseMode.MARKDOWN
        )

        # Extract & group
        all_images = extract_all_images(zip_path, tmpdir)
        if not all_images:
            await status_msg.edit_text("❌ No images found in the ZIP file.")
            return

        # AI grouping 
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
                    f"⏳ Processing product *{i}/{total}* ({len(images)} photos)…",
                    parse_mode=ParseMode.MARKDOWN
                )

                # Generate description
                caption = await loop.run_in_executor(None, generate_description, images, cfg)

                # Post album to channel (Standard PTB is fine for uploading up to 50MB albums)
                for chunk_start in range(0, len(images), MAX_ALBUM_SIZE):
                    chunk = images[chunk_start:chunk_start + MAX_ALBUM_SIZE]
                    ok = await post_album_to_channel(ctx.bot, chunk, caption if chunk_start == 0 else "")
                    if ok:
                        posted += 1
                    else:
                        failed += 1
                    await asyncio.sleep(2)

                if i < total:
                    await asyncio.sleep(12)

            except RuntimeError as e:
                if "Daily Gemini limit" in str(e):
                    await status_msg.edit_text(f"⛔ *Daily quota exhausted!*\n{str(e)}", parse_mode=ParseMode.MARKDOWN)
                    return
                else:
                    log.error(f"Product {i} failed: {e}")
                    failed += 1
                    continue

        icon = "✅" if failed == 0 else "⚠️"
        await status_msg.edit_text(
            f"{icon} *Done!*\n\n"
            f"📮 Posted: *{posted}* product(s) to channel\n"
            f"❌ Failed: *{failed}*",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        log.exception("Unexpected error")
        await status_msg.edit_text(f"❌ Unexpected error:\n<code>{e}</code>", parse_mode=ParseMode.HTML)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Deps ───────────────────────────────────────────────────────────────────────
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                              CallbackQueryHandler, ContextTypes, filters)
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

# ── Description templates (fixed text blocks, saved by name) ──────────────────
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
                model="gemini-2.5-flash-lite-preview-06-17", contents=parts)
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
    """
    Passes all images to Gemini simultaneously for visual grouping.
    Uses strict JSON keys and explicit visual cues to guarantee sorting order.
    """
    parts = []
    for i, path in enumerate(image_paths):
        parts.append(types.Part.from_text(text=f"Image Index: {i}"))
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
        
    prompt = """
    You are an expert sports apparel merchandiser. Look at all the provided images.

    Group the images that show the EXACT same physical product.

    COLOR IS THE MOST IMPORTANT SIGNAL — READ THIS CAREFULLY:
    - Compare the DOMINANT jersey color of each image first, before anything else.
    - If two images have a clearly different dominant color (e.g. one is green, another is yellow;
      one is white, another is red) they are ALWAYS DIFFERENT PRODUCTS — put them in DIFFERENT groups,
      even if both show the same brand logo (e.g. both have an Adidas logo), same generic style,
      or no visible team crest at all.
    - A visible brand logo (Adidas/Nike/Puma) alone is NEVER enough to group two images together.
      Brand logos repeat across many different unrelated products.
    - Only group images together if you are confident they are PHOTOS OF THE SAME PHYSICAL ITEM —
      same color, same trim/accent color, same pattern — just from a different angle or a
      render vs a real photo of that same item.
    - If you are not sure two images are the same product, DO NOT group them together.
      It is much better to create an extra separate group than to wrongly merge two different products.

    CRITICAL SORTING RULES:
    You must distinguish between "3D Renders" and "Real Photos" based on these visual clues:
    - 3D Renders: Have a pure white background, stand upright (ghost mannequin), and often have the "KOS" logo in the corner.
    - Real Photos: Are flat-lays on a greyish floor/surface, have visible fabric wrinkles, and often show the shirt and shorts laid out side-by-side.
    
    For each product group, identify the image indices for:
    - "front_view": MUST be the Front-facing 3D Render (white background). NEVER put a flat-lay real photo here if a 3D render exists.
    - "back_view": MUST be the Back-facing 3D Render (white background). (Use null if there is no back view).
    - "other_photos": Put ALL Real Photos (flat-lays on grey backgrounds) in this array.
    
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
        
        # Safely extract the JSON array of objects
        import re
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            structured_groups = json.loads(match.group())
        else:
            structured_groups = json.loads(raw)
            
    except Exception as e:
        log.warning(f"Direct clustering failed ({e}), falling back to 1 group per image.")
        return [[p] for p in image_paths]
        
    # Map indices back to file paths enforcing strict Python sorting
    result = []
    for group in structured_groups:
        current_group_paths = []
        
        # 1. Force Front View to absolute position 0
        front_idx = group.get("front_view")
        if isinstance(front_idx, int) and front_idx < len(image_paths):
            current_group_paths.append(image_paths[front_idx])
            
        # 2. Force Back View to absolute position 1
        back_idx = group.get("back_view")
        if isinstance(back_idx, int) and back_idx < len(image_paths) and back_idx != front_idx:
            current_group_paths.append(image_paths[back_idx])
            
        # 3. Add all remaining photos last
        for other_idx in group.get("other_photos", []):
            if isinstance(other_idx, int) and other_idx < len(image_paths) and other_idx not in (front_idx, back_idx):
                current_group_paths.append(image_paths[other_idx])
                
        if current_group_paths:
            result.append(current_group_paths)
            
    return result or [image_paths]

def detect_product_category(image_paths: list) -> str:
    """
    Ask Gemini what TYPE of product this group is, so the caption can use the
    right wording/sizes (jersey vs boots vs other). One cheap call per group.
    Returns one of: "jersey", "boots", "other"
    """
    parts = []
    for path in image_paths[:3]:
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text="""
Look at this product image. What category of sportswear product is it?

Reply with ONLY one word, no punctuation, no explanation:
- "jersey" if it's a football/sports shirt, kit, or jersey+shorts set (worn on the torso)
- "boots" if it's football boots/cleats/sneakers (worn on the feet)
- "other" if it's anything else (gloves, balls, bags, accessories, etc.)
"""))
    try:
        raw = gemini_call(parts).strip().lower()
        raw = re.sub(r'[^a-z]', '', raw)
        if raw in ("jersey", "boots", "other"):
            return raw
    except Exception as e:
        log.warning(f"Category detection failed: {e}")
    return "jersey"  # safe default — matches original behaviour

# ── Per-category caption profiles ──────────────────────────────────────────────
# Each profile defines the example structure Gemini should mimic for that
# product type. "sizes_key" looks up the right size list from shop.json,
# falling back to "sizes" for backward compatibility with existing configs.
CATEGORY_PROFILES = {
    "jersey": {
        "sizes_key": "sizes",
        "default_sizes": "S, M, L, XL, 2XL",
        "example": (
            "Футбольня форма клуба/сборной:\n"
            "⚽Многофункциональная: Идеально подходит для футбола бега и интенсивных тренировок.\n"
            "📐Размеры: В наличии размеры {sizes}.\n"
            "🧵Материал: Легкий и дышащий полиэстер\n"
            "💡Стиль: Спортивный с длинным/коротким рукавом.\n"
            "👕Комфорт: Свободная и удобная посадка для удобства в движении.\n"
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
            "⚽Многофункциональные: Идеально подходят для игры на натуральном и искусственном газоне.\n"
            "📐Размеры: В наличии размеры {sizes}.\n"
            "🧵Материал: Прочный и легкий верх, обеспечивающий контроль мяча.\n"
            "💡Стиль: Обтекаемый дизайн, эластичный язычок, мягкий воротник.\n"
            "👟Комфорт: Плотная посадка для уверенного контроля на поле.\n"
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
            "⚽Многофункциональный: Идеально подходит для тренировок и игр.\n"
            "📐Размеры: {sizes}.\n"
            "🧵Материал: Качественные материалы.\n"
            "🚚Доставка осуществляется в течении {delivery}\n"
            "💰{price}\n\n"
            "{uzum_line}{tg_line}{admin}"
        ),
    },
}

def build_caption_prompt(cfg: dict, category: str = "jersey") -> str:
    lang_map  = {"ru": "Russian", "uz": "Uzbek", "en": "English"}
    lang      = lang_map.get(cfg.get("language", "ru"), "Russian")
    uzum_line = f"Наш магазин на Узум:\n{cfg['uzum_link']}\n" if cfg.get("uzum_link") else ""
    tg_line   = f"Telegram: {cfg['telegram_link']}\n"          if cfg.get("telegram_link") else ""
    admin     = cfg.get("admin_tag", "")
    price     = cfg.get("price", "")
    delivery  = cfg.get("delivery", "")

    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["jersey"])
    sizes   = cfg.get(profile["sizes_key"]) or cfg.get("sizes") or profile["default_sizes"]

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
        f"- Include: purpose, sizes ({sizes}), material, style, comfort, delivery ({delivery}), price ({price})\n"
        f"- Use wording appropriate for this exact product type — do not call boots a 'jersey' or mention sleeves on footwear, etc.\n"
        f"- End with shop links and admin tag exactly as shown\n"
        f"- Output ONLY the post text, no markdown, no backticks"
    )


def generate_description(image_paths: list, cfg: dict) -> str:
    category = detect_product_category(image_paths)
    log.info(f"  Detected category: {category}")
    parts = []
    for path in image_paths[:4]:
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=build_caption_prompt(cfg, category)))
    return gemini_call(parts)

# ── Telegram posting ───────────────────────────────────────────────────────────
async def post_album_to_channel(bot, image_paths: list, caption: str) -> bool:
    """Post a photo album to the channel."""
    import aiohttp as _aio
    base  = f"https://api.telegram.org/bot{BOT_TOKEN}"
    media = []
    files = {}

    for i, path in enumerate(image_paths):
        field = f"photo{i}"
        files[field] = open(path, "rb")
        entry = {"type": "photo", "media": f"attach://{field}"}
        if i == 0:
            entry["caption"]    = caption
            entry["parse_mode"] = "HTML"
        media.append(entry)

    try:
        async with _aio.ClientSession() as session:
            form = _aio.FormData()
            form.add_field("chat_id", str(CHANNEL_ID))
            form.add_field("media",   json.dumps(media))
            for field, fobj in files.items():
                form.add_field(field, fobj, filename=Path(fobj.name).name,
                               content_type="image/jpeg")
            async with session.post(f"{base}/sendMediaGroup", data=form) as resp:
                result = await resp.json()
                return result.get("ok", False)
    finally:
        for fobj in files.values():
            fobj.close()

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
        f"Just send me a *ZIP file* with product photos — I'll auto-group them by product, "
        f"write a description for each, then let you pick a style/template before posting.\n\n"
        f"Commands:\n"
        f"/start — show this message\n"
        f"/quota — check remaining Gemini quota\n"
        f"/shop — show current shop config\n"
        f"/newtemplate Name | text — save a reusable text block\n"
        f"/templates — list saved templates\n"
        f"/deltemplate Name — delete a template",
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

async def cmd_newtemplate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Usage: /newtemplate Template Name | template text goes here"""
    if not is_allowed(update.effective_user.id): return
    raw = update.message.text.partition(" ")[2].strip()
    if "|" not in raw:
        await update.message.reply_text(
            "📝 *Usage:*\n`/newtemplate Name | template text here`\n\n"
            "Example:\n`/newtemplate Black Friday | 🔥 Скидка 20%! Только сегодня!`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    name, _, body = raw.partition("|")
    name = name.strip()
    body = body.strip()
    if not name or not body:
        await update.message.reply_text("⚠️ Both a name and template text are required.")
        return

    templates = load_templates()
    templates[name] = body
    save_templates(templates)
    await update.message.reply_text(
        f"✅ Template saved as *{name}*\n\n{body}",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_templates(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    templates = load_templates()
    if not templates:
        await update.message.reply_text(
            "📭 No templates saved yet.\nCreate one with:\n"
            "`/newtemplate Name | template text`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    lines = [f"• *{name}*\n  {body[:80]}{'…' if len(body) > 80 else ''}"
             for name, body in templates.items()]
    await update.message.reply_text(
        "📋 *Saved templates:*\n\n" + "\n\n".join(lines),
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_deltemplate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Usage: /deltemplate Name"""
    if not is_allowed(update.effective_user.id): return
    name = update.message.text.partition(" ")[2].strip()
    templates = load_templates()
    if name not in templates:
        await update.message.reply_text(f"⚠️ No template named *{name}* found.", parse_mode=ParseMode.MARKDOWN)
        return
    del templates[name]
    save_templates(templates)
    await update.message.reply_text(f"🗑 Deleted template *{name}*", parse_mode=ParseMode.MARKDOWN)

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
        templates = load_templates()

        # ── Ask for the template ONCE for the whole batch ──────────────────
        batch_key = f"{update.effective_chat.id}_{update.message.message_id}"
        buttons = [[InlineKeyboardButton("🚫 No template — post as is",
                                          callback_data=f"batchtmpl::__none__::{batch_key}")]]
        for name in templates.keys():
            payload = f"batchtmpl::{name}::{batch_key}"
            if len(payload.encode("utf-8")) > 64:
                continue
            buttons.append([InlineKeyboardButton(f"📋 {name}", callback_data=payload)])
        keyboard = InlineKeyboardMarkup(buttons)

        event = asyncio.Event()
        ctx.bot_data.setdefault("pending_batches", {})[batch_key] = {
            "event": event,
            "template": None,
        }

        await status_msg.edit_text(
            f"📦 *{doc.file_name}*\n"
            f"🖼 Found *{total} product(s)*\n\n"
            f"Choose a description style to apply to *all {total} products* in this batch:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )

        await event.wait()
        chosen = ctx.bot_data["pending_batches"].pop(batch_key, {})
        tmpl_name = chosen.get("template") or "__none__"
        tmpl_text = templates.get(tmpl_name, "") if tmpl_name != "__none__" else ""

        await status_msg.edit_text(
            f"📦 *{doc.file_name}*\n"
            f"🖼 *{total} product(s)* — style: *{tmpl_name if tmpl_name != '__none__' else 'No template'}*\n"
            f"📊 Quota: {rate_limiter.remaining_today}/{GEMINI_RPD} left\n\n"
            f"⏳ Starting to post…",
            parse_mode=ParseMode.MARKDOWN
        )

        # ── Generate + post every product automatically with the chosen style ──
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

                caption = await loop.run_in_executor(None, generate_description, images, cfg)
                if tmpl_text:
                    caption = f"{caption}\n\n{tmpl_text}"

                ok_all = True
                for chunk_start in range(0, len(images), MAX_ALBUM_SIZE):
                    chunk = images[chunk_start:chunk_start + MAX_ALBUM_SIZE]
                    ok = await post_album_to_channel(ctx.bot, chunk,
                                                     caption if chunk_start == 0 else "")
                    ok_all = ok_all and ok
                    await asyncio.sleep(2)

                if ok_all:
                    posted += 1
                else:
                    failed += 1

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

# ── Template selection callback (applies to the whole ZIP batch) ──────────────
async def handle_template_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not is_allowed(query.from_user.id):
        return

    try:
        _, tmpl_name, batch_key = query.data.split("::", 2)
    except ValueError:
        await query.edit_message_text("⚠️ This button has expired.")
        return

    pending = ctx.bot_data.get("pending_batches", {}).get(batch_key)
    if not pending:
        await query.edit_message_text("⚠️ This session has expired (bot may have restarted). Please resend the ZIP.")
        return

    pending["template"] = tmpl_name
    event = pending.get("event")
    if event:
        event.set()
    # The main handle_zip loop takes over from here and edits status_msg itself.

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting Smart Product Poster bot…")

    if not load_shop_config():
        log.warning("⚠️  shop.json not found — bot will warn users until it's created.")

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

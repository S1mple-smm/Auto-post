#!/usr/bin/env python3
"""
Telegram ZIP Bot
- Auto-groups photos by product using Gemini Vision
- Strict Album Sorting: Custom order for Jerseys vs Boots
- Caps albums at 10 images, skips excess
- Excludes color/season from generated titles
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
API_ID         = int(os.environ["TELEGRAM_API_ID"])      
API_HASH       = os.environ["TELEGRAM_API_HASH"]        
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
def load_image_bytes(path: str, max_size=(512, 512)) -> bytes:
    """Downscaled to 512x512 so massive batches (26+ images) won't timeout."""
    with PILImage.open(path) as img:
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return buf.getvalue()

def safe_int(val):
    """Safely converts stringified JSON numbers to ints so they aren't ignored."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def gemini_call(parts: list, max_retries: int = 5, is_json: bool = False) -> str:
    rate_limiter.wait_if_needed()
    client = genai.Client(api_key=GEMINI_KEY)
    
    # Disable safety filters (sports mannequins often trigger false positives)
    config = types.GenerateContentConfig(
        temperature=0.1,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )
    if is_json:
        config.response_mime_type = "application/json"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite", 
                contents=parts,
                config=config
            )
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

def ai_group_images(image_paths: list) -> tuple:
    """Returns (grouped_lists, error_message). Uses highly structured JSON roles for exact sorting."""
    parts = []
    for i, path in enumerate(image_paths):
        parts.append(types.Part.from_text(text=f"Image Index: {i}"))
        parts.append(types.Part.from_bytes(data=load_image_bytes(path), mime_type="image/jpeg"))
        
    prompt = """
    You are an expert sportswear authenticator. Group the provided images by EXACT physical product.

    CRITICAL INSTRUCTION: You are receiving many images, but there are only a FEW distinct products. DO NOT put every image in its own group! Combine all angles, renders, and photos of the exact same item into ONE group.

    GROUPING RULES:
    1. DIFFERENT MODELS = DIFFERENT GROUPS.
    2. DIFFERENT BASE COLORS = DIFFERENT GROUPS.
    3. DIFFERENT LOGO/ACCENT COLORS = DIFFERENT GROUPS (e.g., beige boot with GREEN logo is different from beige boot with GOLD logo).
    4. BOOT COLLAR: A boot WITH an ankle sock is a completely different product from a boot WITHOUT a sock.

    IMAGE CATEGORIZATION (Assign every index to EXACTLY ONE field):
    - "product_type": "boots" or "jersey"
    - "infographics": [indices of images with text, specs, Russian words, grass icons, or collages]
    - "render_front": index of front 3D render (white background, no text). null if none.
    - "render_back": index of back/side 3D render. null if none.
    - "photo_front": index of real photo front view (on person or clean profile). null if none.
    - "photo_back": index of real photo back/heel view. null if none.
    - "photo_bottom_sole": index of bottom spikes/sole. null if none.
    - "photo_closeups": [indices of zoomed-in details like texture, logo, laces]
    - "photo_flatlays": [indices of item laid flat on floor/table]

    For each group, provide "model_analysis" describing base color, logo color, sole, and collar type.

    EVERY SINGLE IMAGE INDEX MUST APPEAR EXACTLY ONCE. ALL INDICES MUST BE INTEGERS, NOT STRINGS.
    Return ONLY a valid JSON array of objects.

    EXAMPLE FORMAT:
    [
      {
        "model_analysis": "Beige Phantom, no sock, green swoosh",
        "product_type": "boots",
        "infographics": [0, 5],
        "render_front": 1,
        "render_back": 2,
        "photo_front": null,
        "photo_back": null,
        "photo_bottom_sole": 3,
        "photo_closeups": [4],
        "photo_flatlays": []
      }
    ]
    """
    parts.append(types.Part.from_text(text=prompt))
    
    log.info(f"Sending {len(image_paths)} images to Gemini for structural clustering...")
    error_msg = None
    
    try:
        raw = gemini_call(parts, is_json=True)
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        structured_groups = json.loads(raw)
    except Exception as e:
        log.warning(f"Clustering failed ({type(e).__name__}: {e}).")
        error_msg = f"AI Grouping Failed: {e}"
        return [[p] for p in image_paths], error_msg
        
    result = []
    seen_indices = set()
    for group in structured_groups:
        ptype = group.get("product_type", "jersey").lower()
        
        # Safely convert AI outputs
        front_r = safe_int(group.get("render_front"))
        back_r = safe_int(group.get("render_back"))
        front_p = safe_int(group.get("photo_front"))
        back_p = safe_int(group.get("photo_back"))
        bottom_p = safe_int(group.get("photo_bottom_sole"))
        
        raw_info = group.get("infographics", [])
        raw_close = group.get("photo_closeups", [])
        raw_flat = group.get("photo_flatlays", [])
        
        infos = [safe_int(x) for x in (raw_info if isinstance(raw_info, list) else []) if safe_int(x) is not None]
        closeups = [safe_int(x) for x in (raw_close if isinstance(raw_close, list) else []) if safe_int(x) is not None]
        flatlays = [safe_int(x) for x in (raw_flat if isinstance(raw_flat, list) else []) if safe_int(x) is not None]

        # Gather all mentioned indices so we don't lose any
        all_mentioned = []
        for idx in [front_r, back_r, front_p, back_p, bottom_p] + infos + closeups + flatlays:
            if idx is not None:
                all_mentioned.append(idx)
                seen_indices.add(idx)

        current_group_paths = []
        def add_to_group(idx):
            if idx is not None and 0 <= idx < len(image_paths) and image_paths[idx] not in current_group_paths:
                current_group_paths.append(image_paths[idx])

        # === THE EXACT SORTING LOGIC REQUESTED ===
        if ptype == "boots":
            # BOOTS: 1. Infographic, 2. Front, 3. Back, 4. Bottom
            if infos: add_to_group(infos[0])
            add_to_group(front_r)
            add_to_group(front_p)
            add_to_group(back_r)
            add_to_group(back_p)
            add_to_group(bottom_p)
            for i in infos[1:]: add_to_group(i)
            for i in closeups: add_to_group(i)
            for i in flatlays: add_to_group(i)
            for i in all_mentioned: add_to_group(i) # catch-all for any stragglers
        else:
            # JERSEY: 1. Render Front, 2. Render Back, 3. Photo Front, 4. Photo Back, 5. Closeups, 6. Flatlays (Floor)
            add_to_group(front_r)
            add_to_group(back_r)
            add_to_group(front_p)
            add_to_group(back_p)
            for i in closeups: add_to_group(i)
            for i in flatlays: add_to_group(i)
            for i in infos: add_to_group(i)
            for i in all_mentioned: add_to_group(i) # catch-all for any stragglers
            
        # Strictly discard anything over MAX_ALBUM_SIZE (10)
        if current_group_paths:
            result.append(current_group_paths[:MAX_ALBUM_SIZE])

    # Failsafe: if the AI forgot any images, put them in their own groups
    missing = [i for i in range(len(image_paths)) if i not in seen_indices]
    if missing:
        for idx in missing:
            result.append([image_paths[idx]])

    return result or [[p] for p in image_paths], error_msg

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
        "title_rule": "First line MUST be 'Футбольная форма [Название команды]:' on a single line. CRITICAL LANGUAGE RULE: Write Club names in ENGLISH (e.g., 'Футбольная форма Real Madrid:'). Write National Team/Country names in RUSSIAN (e.g., 'Футбольная форма Сборной Португалии:'). DO NOT put the team name on a separate line.",
        "example": (
            "Футбольная форма Сборной Португалии:\n"
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
        "title_rule": "First line MUST be 'Футбольные бутсы [Название модели]:' on a single line. CRITICAL LANGUAGE RULE: Write the model name strictly in ENGLISH (e.g., 'Футбольные бутсы Phantom:', 'Футбольные бутсы Mercurial:'). IDENTIFY the specific model name ONLY. DO NOT include colors, seasons, years, or parent brand names.",
        "example": (
            "Футбольные бутсы Phantom:\n"
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
        "title_rule": "First line MUST be '[Название аксессуара]:' on a single line (e.g., 'Спортивный рюкзак:' or 'Вратарские перчатки:').",
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
    title_rule = profile.get("title_rule", "First line: Identify the product clearly on a single line.")

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
            f"2. TITLE STRUCTURE: {title_rule}\n"
            f"3. STRICT OMISSIONS: DO NOT include the color of the item. DO NOT include the season or year (e.g. '2024'). DO NOT use parent brand names (like 'Adidas', 'Nike').\n"
            f"4. Ensure shop details are injected (Sizes: {sizes}, Price: {price}, Delivery: {delivery}).\n"
            f"5. Always append the shop links at the very bottom:\n{uzum_line}{tg_line}{admin}\n"
            f"6. Output ONLY ONE final completed template."
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
            f"- TITLE STRUCTURE: {title_rule}\n"
            f"- STRICT OMISSIONS: DO NOT include colors, seasons, years, or parent brand names.\n"
            f"- Emoji bullets for each feature\n"
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
        await telethon_client.start(bot_token=BOT_TOKEN)
        telethon_msg = await telethon_client.get_messages(update.effective_chat.id, ids=update.message.message_id)
        await telethon_client.download_media(telethon_msg.media, file=zip_path)

        await status_msg.edit_text(f"🤖 Analyzing photos and grouping...", parse_mode=ParseMode.MARKDOWN)

        all_images = extract_all_images(zip_path, tmpdir)
        loop = asyncio.get_event_loop()
        groups, ai_error = await loop.run_in_executor(None, ai_group_images, all_images)

        if ai_error:
            await update.message.reply_text(f"⚠️ **AI Grouping Notice:**\n`{ai_error}`", parse_mode=ParseMode.MARKDOWN)

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

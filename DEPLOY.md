# Deployment Guide — Smart Product Poster Bot

## How it works
1. You send a ZIP file to the bot in private chat
2. Bot auto-groups photos by product using Gemini AI
3. Posts each product as a separate album to your channel
4. Reports progress back to you

---

## Option A — Deploy on Railway (easiest, free tier available)

1. Push this folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add environment variables in Railway dashboard:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `GEMINI_API_KEY`
   - `ALLOWED_USER_IDS` (your Telegram user ID)
4. Railway auto-builds from Dockerfile and keeps it running 24/7

---

## Option B — Deploy on any VPS (Ubuntu)

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh

# 2. Upload files to server
scp -r ./smart_poster user@your-server:/home/user/

# 3. SSH into server
ssh user@your-server
cd smart_poster

# 4. Create .env file
cp .env.example .env
nano .env   # fill in your values

# 5. Run
docker compose up -d

# View logs
docker compose logs -f
```

---

## Option C — Run directly with Python (no Docker)

```bash
pip install python-telegram-bot google-genai Pillow aiohttp

export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
export GEMINI_API_KEY="..."
export ALLOWED_USER_IDS="your_telegram_user_id"

python telegram_bot.py
```

---

## Get your Telegram User ID
Message @userinfobot on Telegram — it replies with your user ID number.
Put this in ALLOWED_USER_IDS so only you can use the bot.

---

## Bot commands
| Command | Description |
|---------|-------------|
| `/start` | Show welcome + quota info |
| `/quota` | Check remaining Gemini requests today |
| `/shop`  | Show current shop config |

## Usage
Just send a ZIP file to the bot in private chat — it handles everything automatically.

---

## shop.json
Place this file next to telegram_bot.py (or mount it via Docker volume):

```json
{
  "shop_name": "Kings of Sport",
  "uzum_link": "https://uzum.uz/shop/kingsofsport",
  "telegram_link": "https://t.me/kings_of_sport",
  "admin_tag": "@kos_sport_admin",
  "price": "320.000 сум",
  "delivery": "2-3 часа по Ташкенту / 2-3 дня по Областям",
  "sizes": "S, M, L, XL, 2XL",
  "language": "ru"
}
```

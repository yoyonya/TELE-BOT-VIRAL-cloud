import os
from dotenv import load_dotenv
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
)

from telegram.error import NetworkError, BadRequest

from query import ask, TOPICS, LAYERS_EXPLANATION

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")


# ------------------------------------------------
# SAFE MESSAGE SPLITTER
# ------------------------------------------------

MAX_LEN = 4000  # rezerva pod Telegram limitem 4096


async def send_long_message(update, text: str):

    if not text:
        return

    for i in range(0, len(text), MAX_LEN):
        chunk = text[i:i + MAX_LEN]

        try:
            await update.message.reply_text(chunk)

        except BadRequest:
            # fallback kdyby Telegram protestoval
            await update.message.reply_text(chunk[:3900])


# ------------------------------------------------
# COMMANDS
# ------------------------------------------------

async def topics_command(update, context):
    await send_long_message(update, TOPICS)


async def layers_command(update, context):
    await send_long_message(update, LAYERS_EXPLANATION)


# ------------------------------------------------
# MESSAGES
# ------------------------------------------------

async def handle_message(update, context):

    question = update.message.text

    if not question:
        return

    # ignoruj slash (bez pádu handleru)
    if question.startswith("/"):
        return

    try:

        answer = ask(question)

    except Exception as e:

        print("BOT ERROR:", e)

        answer = """
Model dočasně neodpovídá.

Zkus dotaz zopakovat.
"""

    await send_long_message(update, answer)


# ------------------------------------------------
# GLOBAL ERROR HANDLER (VELMI DŮLEŽITÉ)
# ------------------------------------------------

async def error_handler(update, context):

    print("TELEGRAM ERROR:", context.error)

    if isinstance(context.error, NetworkError):
        print("Network glitch — bot pokračuje.")

    elif isinstance(context.error, BadRequest):
        print("Bad request — pravděpodobně velikost zprávy.")


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    print("▶ Starting epistemic bot")

    app = Application.builder().token(TOKEN).build()

    # commands
    app.add_handler(CommandHandler("topics", topics_command))
    app.add_handler(CommandHandler("layers", layers_command))

    # messages
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # error handler (většina botů ho nemá — velká chyba)
    app.add_error_handler(error_handler)

    print("▶ Bot is running")

    app.run_polling()


if __name__ == "__main__":
    main()

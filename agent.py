import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Chybí GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def run(prompt: str) -> str:
    if not prompt.strip():
        return "[prázdný vstup – nic neposílám]"
    response = client.models.generate_content(
        model = 'models/gemini-3-pro-preview',
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            break
        print(run(user_input))

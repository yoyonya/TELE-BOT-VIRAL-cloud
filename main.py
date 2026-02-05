# main.py

from query import ask


def main():
    print("UX / Epistemický BOT – napiš otázku (exit = konec)")
    print("──────────────────────────────────────────────")

    while True:
        q = input("\nOtázka: ").strip()

        if q.lower() in ["exit", "quit"]:
            break

        answer = ask(q)

        print("\nOdpověď:\n")
        print(answer)


if __name__ == "__main__":
    main()

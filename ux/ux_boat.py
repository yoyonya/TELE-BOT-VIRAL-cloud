def ux_preprocess(question: str):
    q = question.strip()

    # default: nedělej nic
    result = {
        "question": q,
        "note": None
    }

    # příliš obecné / emoční
    if len(q.split()) < 3:
        result["note"] = (
            "Poznámka: Dotaz je velmi obecný. "
            "Systém pracuje lépe s otázkami typu "
            "„jaké vzorce“, „co bylo pozorováno“, „co nevíme“."
        )

    if any(x in q.lower() for x in [
        "co mám dělat", "mám to vzít", "poradíš", "pomoz"
    ]):
        result["note"] = (
            "Poznámka: Tento nástroj neposkytuje rady. "
            "Odpovídá pouze popisem pozorovaných jevů a vzorců."
        )

    return result


def ux_help():
    return (
        "Tento nástroj odpovídá pouze na otázky typu:\n\n"
        "• co bylo pozorováno / zaznamenáno\n"
        "• jaké vzorce se opakují\n"
        "• co nevíme / kde jsou zkreslení\n\n"
        "Příklady:\n"
        "– Jaké události se po prvním virálu objevují nejčastěji?\n"
        "– Jaké reakce lidí se opakují napříč případy?\n"
        "– Co v příbězích o virálním úspěchu systematicky chybí?"
    )

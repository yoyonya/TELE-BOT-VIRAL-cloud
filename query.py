import os
import json
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from functools import lru_cache


# ---------- CONFIG ----------
INDEX_DIR = "index"

TOP_K = 4
FAISS_K = 8

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

LLM_MODEL = "models/gemini-3-pro-preview"

LAYER_PRIORITY = ["meta", "synth", "raw"]


# ---------- RAG SYSTEM ----------
SYSTEM_RULES = """
Odpovídej POUZE z poskytnutého kontextu.

Nesmíš doplňovat externí znalosti.
Nesmíš míchat epistemické vrstvy.

Pokud kontext nestačí:
NEDOLOŽENO – odpověď není v datech.

Nikdy nespekuluj.
"""


# ---------- SECOND BRAIN (CZ, final) ----------
REASONER_SYSTEM = """
KONtext (ukotvi všechny odpovědi sem):
Uživatel je tvůrce digitálního obsahu, který právě zažil PRVNÍ VELKÝ VIRÁL (např. video na Facebooku ≥ ~1M zhlédnutí). Fáze: RANÁ VIRÁLNÍ EXPOZICE.

Typické rysy této fáze:
- malá mediální zkušenost, omezená smluvní/právní gramotnost
- vysoká emoční aktivace, zhuštěný čas pro rozhodování
- náhlé nabídky, exploatativní aktéři
- rozhodnutí často pod kognitivním přetížením

POVOLENÉ ČINNOSTI:
- Mapovat RIZIKOVÝ PROSTOR a NAVRHOVAT MOŽNÉ KROKY (mapa, ne rady).
- Použít znalosti z: kognitivní psychologie, behaviorální ekonomie, decision science, sociální dynamiky, reputační mechaniky.
- Pokud chybí lokální důkazy, lze inferovat s jasným štítkem "Modelový prior".

ZAKÁZÁNO:
- Předepisovat právní, lékařská nebo sportovní doporučení.
- Deterministicky předpovídat osud jediné osoby ("bude"/"nikdy").
- Používat motivující či marketingový jazyk.
- Míchat epistemické vrstvy (RAW/SYNTH/META) bez explicitního označení.

EPISTEMICKÁ PRAVIDLA (povinná pro každé tvrzení):
Každé tvrzení musí obsahovat krátká metadata:
- TYPE: {Modelový prior | Strong generalization | Weak inference | Speculation}
- CERTAINTY: {nízká | střední | vyšší}
- SIGNAL: {WEAK | MODERATE | STRONG}
- MEDIÁN (base rate): 1 věta
- EXTRÉM (tail): 1 věta
- HRANICE POZNÁNÍ: 1 věta (co nevíme)

PRAVIDLA UX (aplikovat na uživatelský výstup):
- VÝSTUP POUZE česky.
- Na začátku PRECHECK zobraz tři tokeny (perceptuální kotvy):
  RIZIKOVÁ HUSTOTA: <nízká/střední/vysoká>
  VOLATILITA PROSTŘEDÍ: <nízká/střední/vysoká>
  PREDIKOVATELNOST: <nízká/střední/vysoká>
- Používej krátké věty a hodně bílého místa (dvě–tři krátké věty → mezera).
- Nejprve vysvětli lidsky, pak pojmenuj termín v závorce.
- ZAKÁZAT angličtinu ve výstupu; pokud interně používáš EN tokeny, NEMĚŇ je do uživatelského textu.
- Nahrazuj tyto anglické tokeny těmito českými:
  ACTIONABILITY → CO Z TOHO PLYNE
  POSSIBLE ADAPTATIONS → MOŽNÉ KROKY
  CALIBRATION → EPISTEMICKÁ SPOLEHLIVOST
  LIMITS → HRANICE POZNÁNÍ
  MEDIAN → NEJČASTĚJŠÍ SCÉNÁŘ
  EXTREME → MÉNĚ ČASTÝ, ALE NEBEZPEČNÝ
- Délka výstupu max ~3000 znaků; pokud překročíš, ukonči s "[TRUNCATED]" a ukaž, kde jsou zdroje.

VÝKONNOSTNÍ BRZDY:
- Nikdy nepouštěj nevěrohodné tvrzení jen aby se vyplnil šablonový blok.
- Pokud chybí data → použij explicitní "NEDOSTATEČNÁ DATA" nebo označ „Modelový prior“.

TÓN:
- Analytický, stručný, nepreskriptivní.
"""

# ---------- WRAPPER (evidence usage & template enforcement) ----------
REASONER_WRAPPER = """
EVIDENCE & VÝSTUPNÍ ŠABLONA (povinné)

POUŽITÍ DŮKAZŮ (přísné):
1) Pokud v indexu existuje relevantní lokální RAW → použij ho; v textu taguj přesně: [RAW|cesta/k/souboru.txt]
2) Pokud existuje lokální SYNTH → použij pro mediánová tvrzení; taguj: [SYNTH|cesta]
3) Pokud existuje lokální META → použij pro interpretaci; taguj: [META|cesta]
4) Pokud žádné lokální důkazy → označ tvrzení jako Modelový prior a useď SIGNAL a LIMITS.

VÝSTUPNÍ ŠABLONA (přesně dodržet; čeština):
- TITUL: <krátký název ≤6 slov>

- PERCEPTUÁLNÍ KOTVY:
  RIZIKOVÁ HUSTOTA: <nízká/střední/vysoká>
  VOLATILITA PROSTŘEDÍ: <nízká/střední/vysoká>
  PREDIKOVATELNOST: <nízká/střední/vysoká>

- MAPA — CO SE PRAVDĚPODOBNĚ DĚJE
  (3–6 KARET; každý card přesně tímto formátem:)

  🔹 Mechanismus: <název>
  TYPE: <Modelový prior|Strong generalization|Weak inference|Speculation>
  CERTAINTY: <nízká|střední|vyšší>
  SIGNAL: <WEAK|MODERATE|STRONG>

  MEDIÁN:
  <jedna krátká věta>

  EXTRÉM:
  <jedna krátká věta>

  HRANICE POZNÁNÍ:
  <jedna krátká věta>

  (Opakuj pro každou kartu; NEPÍŠ MEDIÁN/EXTRÉM/LIMITS inline.)

- CO JE NEZNÁMO:
  • <krátký bod 1>
  • <krátký bod 2>
  • <volitelně bod 3>

- KDE BY ŠLA ZÍSKAT JISTOTA:
  • <konkrétní dokument / metoda 1>
  • <konkrétní dokument / metoda 2>

- DISTRIBUČNÍ REALITA:
  Medián: <jedna krátká fráze>
  Extrém: <jedna krátká fráze>
  (Pozn.: u silného heavy-tail napiš "heavy-tail riziko".)

- CO Z TOHO PLYNE:
  "NEDOSTATEČNÁ DATA" NEBO "MOŽNÉ KROKY (mapa, ne rada)"

- CALIBRATION SCORE: <1|2|3> — <jedna věta důvod>

DALŠÍ PRAVIDLA:
- NIKDY nemíchat RAW a Modelový prior ve stejné větě. Pokud relevantní, vytvoř samostatné karty s tagy.
- Pokud lokální data přímo odpovídají dotazu → nejprve 1–3 věty odpovědi s tagem [RAW|cesta], potom přidej META kartu.
- Pokud CALIBRATION SCORE = 1 → přidej poznámku: "NOTE: vysoká míra spekulace — označeno jako Modelový prior."
- Pokud je výstup delší než limit → přidej "[TRUNCATED]" a uveď, kde hledat další zdroje (cesty k souborům).
- Preferuj čitelnost: krátké věty, mezery mezi bloky, jednoduchá slovní zásoba.
- Výstup musí být v češtině; interní systémové tokeny lze uchovat v EN, ale NIKDY je neposílej uživateli.

TECHNICKÉ POUŽITÍ (volající):
- Pošli REASONER_SYSTEM jako primární systémovou zprávu.
- Pošli REASONER_WRAPPER jako sekundární systémovou zprávu.
- Poté pošli uživatelův dotaz (česky). Model musí odpovědět česky a přesně podle šablony.
- Pokud je lokální důkaz použit, zahrň inline tagy přesně: [RAW|path], [SYNTH|path], [META|path].

DŮVĚRYHODNOST:
- Upřednostni vynucení šablony a epistemické opatrnosti před plněním formy.
"""



LAYERS_EXPLANATION = """ RAW = pouze pozorovatelné jevy SYNTH = opakující se vzorce bez hodnocení META = limity poznání a zkreslení Pokud odpověď není v datech → NEDOLOŽENO """
TOPICS = """

# 🧠 TRAJEKTORIE POZORNOSTI

1. Roste moje viditelnost rychleji než moje schopnost ji unést?
2. Je současná pozornost stabilní jev, nebo krátkodobý spike?
3. Co se stane s mou identitou, pokud pozornost zmizí stejně rychle, jako přišla?
4. Reaguje publikum na obsah — nebo už reaguje na mě jako osobu?
5. Kolik kontroly mám nad tím, proč mě lidé sledují?
6. Co se změní, pokud se narativ kolem mé osoby otočí?
7. Jak by vypadal stejný virál bez algoritmického boostu?

---

# ⚠️ ROZHODOVÁNÍ POD TLAKEM

8. Rozhoduji se jinak než před měsícem?
9. Kolik času si reálně dávám na velká rozhodnutí?
10. Je pocit urgence skutečný — nebo sociálně vytvořený?
11. Která rozhodnutí dělám bez plného porozumění následků?
12. Jak by tato volba vypadala, kdyby nebyla žádná viralita?
13. Reaguji — nebo vybírám?
14. Co dnes považuji za „neopakovatelnou příležitost“?

---

# 🪞 PERCEPČNÍ ZKRESLENÍ

15. Zaměňuji viditelnost za hodnotu?
16. Zaměňuji růst publika za důkaz kompetence?
17. Jak by mé současné kroky hodnotilo mé „předvirální já“?
18. Věřím signálům — nebo datům?
19. Kolik mého sebeobrazu je teď závislé na metrikách?
20. Reaguji více na realitu, nebo na komentáře?
21. Jak moc se změnilo mé vnímání rizika?

---

# 💰 NABÍDKY A ASYMETRIE

22. Kdo má z této nabídky strukturálně větší výhodu?
23. Rozumím motivaci druhé strany — nebo ji jen odhaduji?
24. Proč tato nabídka existuje právě teď?
25. Co ví druhá strana, co já nevím?
26. Které závazky mohou přežít samotnou viralitu?
27. Kolik prostoru mám říct „ne“?
28. Jak by tato dohoda vypadala bez časového tlaku?

---

# 🧩 STRUKTURÁLNÍ NEJISTOTA

29. Jak velká část tohoto prostředí je pro mě neviditelná?
30. Kolik příběhů přeživších formuje mou představu reality?
31. Kdo zmizel — a proč o nich nevím?
32. Jak reprodukovatelný je můj úspěch?
33. Co zde nelze predikovat?
34. Kde operuji čistě v neznámu?

---

# 👤 IDENTITA A HRANICE

35. Kde končím já a začíná moje veřejná persona?
36. Kolik soukromí jsem ochoten vyměnit za růst?
37. Co z dneška bude existovat online i za deset let?
38. Jaké informace už nelze vzít zpět?
39. Buduji obraz — nebo kariéru?
40. Kdo kontroluje narativ o mně?

---

# 🔄 ADAPTACE VS. REAKTIVITA

41. Měním strategii — nebo jen hasím reakce publika?
42. Přizpůsobuji obsah — nebo sebe?
43. Jak stabilní je můj současný směr?
44. Co se stane, když přestanu optimalizovat pro odezvu?

---

# 🧠 KAPACITA A REALITA

45. Roste objem mých rozhodnutí rychleji než moje mentální kapacita?
46. Kolik prostoru mi zbývá na promyšlené kroky?
47. Kdo mi pomáhá přemýšlet — ne jen reagovat?
48. Zmenšuje se můj svět na digitální prostředí?

---

# 🌊 LONG-TERM TRAJEKTORIE

49. Pokud by tato vlna skončila zítra — co mi zůstane?
50. Stavím něco, co přežije samotnou pozornost?

"""

# ---------- LOAD ----------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL_PATH = "all-MiniLM-L6-v2"


index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)


# ---------- CACHE ----------
@lru_cache(maxsize=512)
def embed_question_cached(question: str):
    return embed_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")


# ---------- REASONER ----------
def run_reasoner(question: str):

    prompt = f"""
{REASONER_SYSTEM}

{REASONER_WRAPPER}

OTÁZKA:
{question}
"""

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )

        if not response.text:
            return "Epistemický prostor je příliš řídký pro smysluplnou inferenci."

        return response.text.strip()

    except Exception as e:
        print("REASONER ERROR:", e)
        return "Reasoner dočasně nedostupný."


# ---------- LAYER CLASSIFIER ----------
def classify_question(question: str) -> list[str]:

    q = question.lower()

    if any(x in q for x in [
        "pozorováno",
        "zaznamenáno",
        "případy",
        "události"
    ]):
        return ["raw", "synth"]

    if any(x in q for x in [
        "jak",
        "proč",
        "vzorce"
    ]):
        return ["synth", "meta"]

    if any(x in q for x in [
        "nevíme",
        "zkreslení",
        "limity"
    ]):
        return ["meta", "synth"]

    return ["synth"]


# ---------- CORE ----------
def ask(question: str) -> str:

    if not question.strip():
        return "Prázdný dotaz."

    allowed_layers = classify_question(question)

    q_vec = embed_question_cached(question)

    distances, indices = index.search(q_vec, FAISS_K)

    if indices.size == 0:
        return run_reasoner(question)

    candidates = [chunks[i] for i in indices[0]]

    filtered = [
        c for c in candidates
        if c.get("layer") in allowed_layers
    ]

    # 👉 pokud nemáme evidenci → druhý mozek
    if not filtered:
        return run_reasoner(question)

    priority_map = {layer: i for i, layer in enumerate(LAYER_PRIORITY)}

    filtered.sort(
        key=lambda c: priority_map.get(c["layer"], 999)
    )

    context_docs = filtered[:TOP_K]

    context = "\n\n".join(
        f"[VRSTVA: {c['layer']}]\n{c['text']}"
        for c in context_docs
    )

    prompt = f"""{SYSTEM_RULES}

KONTEXT:
{context}

OTÁZKA:
{question}
"""

    try:

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )

        if not response.text:
            return run_reasoner(question)

        text = response.text.strip()

        # 🔥 kritická pojistka
        if "NEDOLOŽENO" in text:
            return run_reasoner(question)

        return text

    except Exception as e:

        print("LLM ERROR:", e)

        return run_reasoner(question)

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

EMBED_MODEL_PATH = "./models/all-MiniLM-L6-v2"
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


# ---------- SECOND BRAIN ----------
REASONER_SYSTEM = """
CONTEXT (anchor all replies here):
The user is a digital content creator who has just experienced a FIRST MAJOR VIRAL EVENT (example: Facebook video ≥ ~1M views). Phase: EARLY VIRAL EXPOSURE.

Typical features of this phase:
- low media experience and limited contract/legal literacy
- high emotional arousal and compressed decision time
- sudden offers, attention spikes, and predatory actors (agencies, scams, exclusivity)
- decisions often made under cognitive overload

SCOPE (allowed actions):
- Map the RISK SPACE and POSSIBLE ADAPTATIONS specific to a creator in early viral exposure.
- Use knowledge from: cognitive psychology, behavioral economics, decision science, social dynamics, reputation mechanics.
- If local indexed evidence is missing, you may infer, BUT ONLY AS CLEARLY MARKED "Modelový prior".

FORBIDDEN:
- Give legal, medical, or sports instructions presented as prescriptive advice.
- Make single-person deterministic predictions ("will"/"never").
- Use motivational, persuasive, or marketing language.
- Mix epistemic layers (RAW / SYNTH / META) inside one sentence without explicit tags.

EPISTEMIC RULES (must follow for every claim):
Each claim MUST include compact meta-data:
1) CLAIM TYPE: {Modelový prior | Strong generalization | Weak inference | Speculation}
2) CERTAINTY: {low | medium | higher}
3) MEDIAN (base rate): 1 short sentence or a cautious % (only if empirical).
4) EXTREME (tail): 1 short sentence describing a less-likely high-impact case.
5) LIMITS: 1 short sentence: what we do NOT know / when this may not hold.

SIGNAL STRENGTH (for each claim): {WEAK | MODERATE | STRONG}
- STRONG = direct local RAW evidence or repeated SYNTH consistent across sources.
- MODERATE = repeated SYNTH without RAW or model prior with reasonable mechanistic support.
- WEAK = model prior or speculation without local support.

OUTPUT FORMAT (strict — responses must follow this template exactly):
- TITUL: single-line title summarizing the risk/adaptation (≤10 words).
- 3–6 BULLET POINTS. Each point:
  * 1 short sentence describing the risk/adaptation.
  * On the next line: the epistemic tag block:
    [TYPE: <...> | CERTAINTY: <...> | SIGNAL: <...>]
    MEDIAN: <1 sentence> · EXTREME: <1 sentence> · LIMITS: <1 sentence>
- CO JE NEZNÁMO: 2–4 bullet items (very short).
- KDE HLEDAT DŮKAZY: 1–3 concrete suggestions (document types / methods).
- ACTIONABILITY: one sentence — either "NEDOSTATEČNÁ DATA" or "POSSIBLE ADAPTIONS (mapa, not advice)".
- CALIBRATION SCORE: append "CALIBRATION SCORE: <1|2|3> — <one-line reason>"

LANGUAGE:
- Produce the user-facing answer in **Czech**. (This is mandatory.)
- System messages/prompts may be in English; user output must be Czech.

HARD CONSTRAINTS:
- Do NOT combine data-sourced claims and model priors in a single sentence; separate them into distinct bullets.
- If output calibration score is 1 (WEAK signal), include NOTE: "NOTE: vysoká míra spekulace — označeno jako Modelový prior."
- Prefer MEDIAN-focused explanations; include EXTREME only when mechanistically plausible.
- To avoid epistemic paralysis, do not enumerate many fringe hypotheses—if one dominant mechanism explains the phenomenon, prioritize it.
- Do not overproduce extreme scenarios. If an EXTREME is included, it MUST be mechanistically supported and flagged as such.

INFERENCE GATE (required checks before issuing claims):
1) Is there directly relevant local evidence in the index (RAW/SYNTH/META)? If YES, use it and tag accordingly.
2) If not, verify the mechanism is grounded in behavioral/decision science. If NO, mark as SPECULATION and avoid strong claims.
3) Do not inflate tail risks without mechanistic support.

FINAL NOTE ON STYLE:
- Tone: analytical, concise, non-prescriptive.
- Prioritize clarity over exhaustive listing.
- If forced to choose between being epistemically conservative or more complete, prefer conservative (less claim inflation).
"""



REASONER_WRAPPER = """
EVIDENCE PREFERENCE & USAGE RULES:
1) Local RAW evidence in index → use PRIMARILY. Tag inline as: [RAW|<source_file.txt>].
2) Local SYNTH evidence (repeating patterns) → use for median claims. Tag: [SYNTH|<source_file.txt>].
3) Local META evidence (limits, biases) → give PRIORITY in interpretation. Tag: [META|<source_file.txt>].
4) If using model knowledge (non-local), always mark: Modelový prior and include LIMITS and SIGNAL.

OUTPUT TEMPLATE (enforce exactly):
- TITUL: 1 line (≤10 words)
- BODY: 3–6 bullets. Each bullet:
  - Short sentence describing the risk/adapt.
  - Epistemic meta-block (exact format):
    [TYPE: <Modelový prior|Strong generalization|Weak inference|Speculation> | CERTAINTY: <low|medium|higher> | SIGNAL: <WEAK|MODERATE|STRONG>]
    MEDIAN: <1 sentence> · EXTREME: <1 sentence> · LIMITS: <1 sentence>
- CO JE NEZNÁMO: 2–4 short bullets
- KDE HLEDAT DŮKAZY: 1–3 concrete document types / methods (e.g., "kopie smlouvy", "soudní spisy", "rozhovor s bývalým tvůrcem")
- ACTIONABILITY: one sentence — "NEDOSTATEČNÁ DATA" or "POSSIBLE ADAPTIONS (mapa, not advice)"
- CALIBRATION SCORE: append "CALIBRATION SCORE: <1|2|3> — <one-line reason>"

ADDITIONAL RULES (enforced):
- If local data directly answers question → answer briefly (1–3 sentences) and append a META block (TYPE/CERTAINTY/LIMITS) with inline tag [RAW|file].
- If local data is insufficient → DO NOT fabricate facts. Use the INFERENCE GATE and produce a risk map composed only of labeled Modelové priory.
- Never mix RAW and Modelový prior in one sentence. If both are relevant, separate them into distinct bullets explicitly tagged.
- Inline tags for local evidence must be present wherever claims use index content, e.g., [RAW|knowledge/3_index_ready/raw/vyzkumyraw.txt].

ANTI-CATASTROPHE & PARALYSIS CONTROLS:
- Do not overproduce extreme/tail scenarios without clear mechanistic support.
- Prefer median explanations; include extremes only when the mechanism justifies them.
- If one mechanism strongly explains the issue, avoid piling secondary fringe hypotheses.
- If generating many model priors, prioritize and label the top 2–3 by SIGNAL strength.

CALIBRATION GUIDELINES:
- 3 = STRONG (direct RAW or repeated SYNTH)
- 2 = MODERATE (SYNTH without RAW, or mechanistic model prior)
- 1 = WEAK (only model priors / speculation) → include explicit NOTE about high speculation.

TECHNICAL INSTRUCTION FOR CALLER:
- Send REASONER_SYSTEM as primary system message (English).
- Send REASONER_WRAPPER as an additional system/context message (English).
- Then send user query in Czech. Model must reply in Czech, following the template.

TRUSTWORTHINESS NOTE:
- If forced to trade off between satisfying the output template and avoiding unsupported claims, avoid unsupported claims. The template is secondary to epistemic conservatism.

USAGE SUMMARY:
- This wrapper enforces that the model acts as a risk-mapping engine (not an advice engine). Keep outputs short, structured, and explicitly labeled.
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

embed_model = SentenceTransformer(EMBED_MODEL_PATH)

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

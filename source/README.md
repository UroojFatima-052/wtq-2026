# Utility Bill Decoder

Reads K-Electric, LESCO and IESCO electricity bill images, extracts the billing details as JSON (Level 1) and answers customer questions about each bill in plain English (Level 2).

**Runtime:** Python 3.11

## Setup

```
cd source
pip install -r requirements.txt
copy .env.example .env
```
Put your Google AI Studio key in `API_KEY` inside `.env`. (On Mac/Linux use `cp` instead of `copy`.)

## Run

Generate both CSVs from the test templates:
```
python -m billdecoder.run --bills ../test/bills --csv ../test/csv --out ../output
```
Writes `output/level1.csv` and `output/level2.csv`. The last log line confirms row counts and that no cell is empty.

Demo UI (upload a bill, see the extraction, ask questions):
```
streamlit run app.py
```

## AI design: AI reads and explains, code calculates and verifies

**Gemini does what only AI can do**
- Reads tilted photos with Urdu text and grey boxes, across three different bill layouts.
- Understands each question, picks the relevant facts, flags ambiguity, and says when the bill does not show something.
- Writes the explanation in plain English.

**Python does what AI is bad at**
- **Schema cleanup:** removes `Rs.` and commas, `CR` becomes negative, subsidies negative, zero lines dropped, dates validated, only allowed charge and tax types kept, provider taken from the file name. Unreadable values become `null`, never a guess.
- **Guide rules in code:** highest late-payment amount, a single combined tax line becomes `taxes: []`, subtotals never appear as charge lines.
- **Self-check:** the bill is checked against itself. KE line items must add up to the printed subtotals, and LESCO/IESCO Current Bill must equal charges + taxes. On a mismatch, Gemini re-reads the bill with the exact problem, and the new reading is kept only if it is more consistent.
- **All Level 2 maths:** tax share, cost per unit, history averages, months above thresholds, late-payment difference, labelled estimates. Gemini receives these as facts instead of doing arithmetic.
- **Bill quirks:** the current month is excluded from history averages when the chart includes it, and net metering is recognised (negative units = solar export).

**Efficient:** 2 calls per bill (one to read, one for all 12 questions), so 10 test bills use about 20 calls on the free tier.

## Reliability

- Rate limited to stay under 15 requests/minute.
- Retries with backoff on quota, server and network errors, then switches to the fallback allowed model.
- Malformed JSON is retried. Answers the model skipped are re-asked individually.
- If a bill still fails, it gets an all-`null` JSON or a clear "could not be processed" answer, so the run never crashes and no cell is empty.
- **Same results on re-run:** temperature 0, and every reply is cached in `cache/`, so a re-run reproduces identical CSVs. Delete `cache/` to force fresh calls.
- No personal identifiers: prompts forbid them and the schema has no field for them.

## Structure

```
source/
├── billdecoder/
│   ├── config.py    settings from .env
│   ├── llm.py       only place that calls the model: rate limit, retry, fallback, cache
│   ├── extract.py   Level 1: prompt, schema cleanup, self-check
│   ├── answer.py    Level 2: Python-computed facts + answer prompt
│   └── run.py       CLI: CSV templates in, completed CSVs out
├── app.py           Streamlit demo UI
├── requirements.txt
└── .env.example
```

## AI usage

- **LLM used by the code:** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`). Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) is used only as an automatic fallback.
- **APIs / services:** Google AI Studio Gemini API.
- **Libraries:** google-genai, Pillow, python-dotenv, Streamlit.
- **OCR libraries:** none.
- **Prebuilt invoice or receipt extractors:** none.
- **AI tools used to write the code:** Claude (Anthropic) as a coding assistant.

## Limitations

- Level 2 estimates scale the current bill linearly by units and say so, since full slab tariffs are not printed on every bill.
- Very blurry or covered values are returned as `null` by design.
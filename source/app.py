"""Bill Decoder demo UI: upload a bill, see what was extracted, ask questions.

Run:  streamlit run app.py
"""
import html
import json
import tempfile
from pathlib import Path

import streamlit as st

from billdecoder.answer import answer_questions
from billdecoder.extract import extract_bill

st.set_page_config(page_title="Bill Decoder", page_icon="⚡", layout="wide")

GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)' opacity='0.05'/></svg>")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root {{ --ink:#1c1b1a; --muted:#6b675f; --line:#e7e2d8; --paper:#fbfaf6; }}
.stApp {{ background: var(--paper) url("{GRAIN}"); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
html, body, [class*="css"], .stMarkdown, p, label {{ font-family: 'Inter', sans-serif; color: var(--ink); }}
h1, h2, h3, .display {{ font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); letter-spacing: -0.02em; }}

.eyebrow {{ font: 600 12px 'Inter'; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }}
.hero h1 {{ font-size: 56px; line-height: 1; margin: 6px 0 10px; }}
.hero h1 span {{ background: linear-gradient(transparent 62%, #ffe08a 62%); }}
.hero p {{ font-size: 17px; color: var(--muted); max-width: 640px; }}

.pill {{ display:inline-block; padding:4px 12px; border:1.5px solid var(--ink); border-radius:999px;
         font:600 12px 'Inter'; margin-right:6px; background:#fff; }}

.notes {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:18px; margin: 18px 0 8px; }}
.note {{ padding:18px 18px 16px; color:var(--ink); border-radius: 3px 3px 22px 3px;
         box-shadow: 4px 6px 14px -6px rgba(40,30,10,.35); transition: transform .2s; }}
.note:hover {{ transform: rotate(0deg) translateY(-3px) scale(1.02); }}
.note:nth-child(1) {{ background:#fff3b0; transform: rotate(-1.6deg); }}
.note:nth-child(2) {{ background:#d9f0e3; transform: rotate(1.2deg); }}
.note:nth-child(3) {{ background:#fde2d6; transform: rotate(1.4deg); }}
.note:nth-child(4) {{ background:#dfe6fb; transform: rotate(-1deg); }}
.note .k {{ font:600 11px 'Inter'; letter-spacing:.12em; text-transform:uppercase; opacity:.7; }}
.note .v {{ font:700 30px 'Space Grotesk'; margin-top:6px; }}
.note .s {{ font-size:13px; opacity:.75; margin-top:2px; }}

.card {{ background:#fff; border:1.5px solid var(--line); border-radius:14px; padding:18px 20px; margin:14px 0; }}
.card h3 {{ font-size:18px; margin:0 0 12px; }}
.row {{ display:grid; grid-template-columns: 150px 1fr 110px; align-items:center; gap:10px; margin:7px 0; font-size:14px; }}
.bar {{ height:10px; border-radius:6px; background:#f1ede4; overflow:hidden; }}
.bar i {{ display:block; height:100%; border-radius:6px; background:var(--ink); }}
.bar i.tax {{ background:#e07a5f; }} .bar i.neg {{ background:#3f9d6b; }}
.amt {{ text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }}
.dates {{ display:flex; gap:26px; flex-wrap:wrap; font-size:14px; }}
.dates b {{ display:block; font:700 18px 'Space Grotesk'; }}

.answer {{ background:#fff; border-left:5px solid var(--ink); border-radius:4px 14px 14px 4px;
           padding:14px 18px; margin:12px 0; box-shadow: 0 3px 10px -6px rgba(0,0,0,.25); }}
.answer .q {{ font:700 16px 'Space Grotesk'; margin-bottom:6px; }}
.answer .a {{ font-size:15px; line-height:1.55; color:#34322e; }}

[data-testid="stFileUploader"] section {{ background:#fff; border:2px dashed #cfc8b8; border-radius:14px; }}
.stButton button {{ background:var(--ink); color:#fff; border:0; border-radius:999px; padding:10px 26px; font-weight:600; }}
.stButton button p {{ color:#fff !important; font-weight:600; }}
.stButton button:hover {{ background:#000; }}
.stButton button:hover p {{ color:#ffe08a !important; }}
[data-testid="stImage"] img {{ border-radius:10px; box-shadow: 0 10px 30px -12px rgba(0,0,0,.35); }}
</style>
""", unsafe_allow_html=True)


def money(v):
    if v is None:
        return "Not shown"
    if v < 0:
        return f"PKR {abs(v):,.2f}".rstrip("0").rstrip(".") + " CR"
    return f"PKR {v:,.2f}".rstrip("0").rstrip(".")


def esc(text):
    return html.escape(str(text))


LABELS = {"energy": "Energy", "fixed": "Fixed charges", "fpa": "Fuel adjustment", "quarterly_adjustment": "Quarterly adj.",
          "surcharge": "Surcharge", "meter_rent": "Meter rent", "subsidy": "Subsidy", "other": "Other charge",
          "gst": "Sales tax", "electricity_duty": "Electricity duty", "income_tax": "Income tax",
          "municipal_tax": "Municipal tax", "other_tax": "Other tax"}


# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Women Tech Quest 2026 · Build with AI</div>
  <h1>Bill <span>Decoder</span></h1>
  <p>Upload a K-Electric, LESCO or IESCO electricity bill. Get every number pulled out, checked,
     and explained in plain English.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop a bill image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.markdown('<div class="card" style="text-align:center;color:#6b675f">'
                'AI reads the bill · Python checks every number · AI explains it back to you</div>',
                unsafe_allow_html=True)
    st.stop()

# Keep the original file name: the bill id (e.g. KESC_0008) tells us the provider
tmp_dir = Path(tempfile.gettempdir()) / "billdecoder"
tmp_dir.mkdir(exist_ok=True)
image_path = tmp_dir / uploaded.name
image_path.write_bytes(uploaded.getvalue())

if st.session_state.get("bill") != uploaded.name:
    with st.spinner("Reading the bill..."):
        st.session_state.level1, st.session_state.details = extract_bill(image_path)
        st.session_state.bill = uploaded.name
        st.session_state.pop("answers", None)
l1 = st.session_state.level1

left, right = st.columns([5, 6], gap="large")

with left:
    st.image(str(image_path), use_container_width=True)

with right:
    st.markdown(
        f'<span class="pill">{esc(l1.get("provider") or "Unknown")}</span>'
        f'<span class="pill">Tariff {esc(l1.get("tariff") or "n/a")}</span>'
        f'<span class="pill">{esc(l1.get("bill_month") or "")}</span>'
        f'<span class="pill">{esc(l1.get("sanctioned_load_kw") or "?")} kW load</span>',
        unsafe_allow_html=True)

    late = l1.get("payable_after_due_date")
    within = l1.get("payable_within_due_date")
    extra = f"+{money(late - within)} if late" if None not in (late, within) else "Late amount not shown"
    st.markdown(f"""
    <div class="notes">
      <div class="note"><div class="k">Pay by due date</div><div class="v">{money(within)}</div>
           <div class="s">Due {esc(l1.get("due_date") or "n/a")}</div></div>
      <div class="note"><div class="k">Units used</div><div class="v">{esc(l1.get("units_consumed") if l1.get("units_consumed") is not None else "n/a")}</div>
           <div class="s">{esc(l1.get("previous_reading"))} → {esc(l1.get("current_reading"))}</div></div>
      <div class="note"><div class="k">This month's bill</div><div class="v">{money(l1.get("current_bill"))}</div>
           <div class="s">Arrears {money(l1.get("arrears"))}</div></div>
      <div class="note"><div class="k">After due date</div><div class="v">{money(late)}</div>
           <div class="s">{esc(extra)}</div></div>
    </div>""", unsafe_allow_html=True)

    items = [(c, "neg" if c["amount"] < 0 else "") for c in (l1.get("charges") or [])]
    items += [(t, "tax") for t in (l1.get("taxes") or [])]
    if not l1.get("taxes") and l1.get("total_taxes") is not None:
        items.append(({"type": "Taxes (combined)", "amount": l1["total_taxes"]}, "tax"))
    biggest = max([abs(i["amount"]) for i, _ in items] or [1])
    rows = "".join(
        f'<div class="row"><span>{esc(LABELS.get(i["type"], i["type"]))}</span>'
        f'<div class="bar"><i class="{cls}" style="width:{abs(i["amount"]) / biggest * 100:.0f}%"></i></div>'
        f'<span class="amt">{money(i["amount"])}</span></div>'
        for i, cls in items)
    st.markdown(f"""
    <div class="card"><h3>Where your money goes</h3>{rows or "<i>No line items printed</i>"}
      <div class="row" style="border-top:1px solid #e7e2d8;padding-top:8px;margin-top:10px">
        <b>Charges subtotal</b><span></span><span class="amt">{money(l1.get("total_charges"))}</span></div>
      <div class="row"><b>Taxes subtotal</b><span></span><span class="amt">{money(l1.get("total_taxes"))}</span></div>
    </div>
    <div class="card"><h3>Key dates</h3><div class="dates">
      <div>Reading<b>{esc(l1.get("reading_date") or "n/a")}</b></div>
      <div>Issued<b>{esc(l1.get("issue_date") or "n/a")}</b></div>
      <div>Due<b>{esc(l1.get("due_date") or "n/a")}</b></div></div></div>
    """, unsafe_allow_html=True)

    with st.expander("Level 1 JSON (exact output written to level1.csv)"):
        st.code(json.dumps(l1, indent=2, ensure_ascii=False), language="json")

# ---------- Questions ----------
st.markdown('<div class="eyebrow" style="margin-top:28px">Level 2</div>'
            '<h2 style="margin-top:4px">Ask about this bill</h2>', unsafe_allow_html=True)
default_qs = ("How much of my bill is taxes?\n"
              "How does this month compare to my average usage?\n"
              "How much extra will I pay if I pay after the due date?")
text = st.text_area("One question per line", default_qs, height=110, label_visibility="collapsed")

if st.button("Explain my bill"):
    questions = [(f"Q{i}", q.strip()) for i, q in enumerate(text.splitlines(), 1) if q.strip()]
    with st.spinner("Checking the numbers and writing answers..."):
        st.session_state.answers = (questions, answer_questions(image_path, l1, st.session_state.details, questions))

if st.session_state.get("answers"):
    questions, answers = st.session_state.answers
    for qid, q in questions:
        st.markdown(f'<div class="answer"><div class="q">{esc(q)}</div>'
                    f'<div class="a">{esc(answers[qid])}</div></div>', unsafe_allow_html=True)
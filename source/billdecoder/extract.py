"""Level 1: read a bill image and return the required JSON.

One model call per bill. The model reads the image; then plain Python
cleans every value (numbers, CR credits, dates, allowed types) so the
output always matches the schema, even if the model slips.
"""
import logging
import re
from pathlib import Path

from .llm import LLMError, ask_json, load_image

log = logging.getLogger(__name__)

CHARGE_TYPES = {"energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"}
TAX_TYPES = {"gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"}
PROVIDER_FROM_FILE = {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}

NUMBER_FIELDS = ["sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed",
                 "total_charges", "total_taxes", "current_bill", "arrears",
                 "payable_within_due_date", "payable_after_due_date"]
DATE_FIELDS = ["reading_date", "issue_date", "due_date"]
FIELD_ORDER = ["provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date", "issue_date",
               "due_date", "previous_reading", "current_reading", "units_consumed", "charges",
               "total_charges", "taxes", "total_taxes", "current_bill", "arrears",
               "payable_within_due_date", "payable_after_due_date"]

EXTRACT_PROMPT = """You are an expert at reading Pakistani electricity bills (K-Electric, LESCO, IESCO).
The image is one bill. Parts may be in Urdu, tilted, or covered by grey boxes.

Return ONE JSON object with two keys: "level1" and "details".

"level1" must have exactly these fields:
{
 "provider": "KE" | "LESCO" | "IESCO",
 "tariff": string, tariff code exactly as printed (e.g. "A1-R", "A-1a(01)"),
 "sanctioned_load_kw": number (Sanc Load / S.Load / San Load),
 "bill_month": "YYYY-MM",
 "reading_date": "YYYY-MM-DD", "issue_date": "YYYY-MM-DD",
 "due_date": "YYYY-MM-DD" (last date to pay WITHOUT late surcharge),
 "previous_reading": number, "current_reading": number,
 "units_consumed": number (units billed this month, as printed),
 "charges": [{"type": "energy|fixed|fpa|quarterly_adjustment|surcharge|meter_rent|subsidy|other", "amount": number}],
 "total_charges": number or null (printed charges subtotal),
 "taxes": [{"type": "gst|electricity_duty|income_tax|municipal_tax|other_tax", "amount": number}],
 "total_taxes": number or null (printed tax subtotal),
 "current_bill": number (printed amount for this month's bill),
 "arrears": number or null (previous dues carried forward, keep sign),
 "payable_within_due_date": number,
 "payable_after_due_date": number (if several late amounts are shown, the HIGHEST),
 "payable_after_due_date_all": [every TOTAL amount printed as payable after the due date, e.g. both "till 24-Apr" and "after 24-Apr" amounts; NOT the surcharge amounts themselves]
}
}

Rules:
- Copy values AS PRINTED. Do not recalculate, round or fix anything.
- Amounts in PKR as plain numbers: "Rs. 19,964" -> 19964. A credit like "38209CR" -> -38209.
- Subsidy / relief lines go in charges with type "subsidy" and a NEGATIVE amount.
- charges: one entry per printed line in the charges section. Skip blank lines and lines printed as 0.
  Keep separate entries when a type repeats (e.g. two energy slab lines = two "energy" entries).
  energy = cost of units (Variable Upto X Units, Cost of Electricity, Energy Charges). fpa = FPA or FCA.
  quarterly_adjustment = QTA / Uniform Quarterly Adjustment. surcharge = e.g. Additional Surcharge (PHL), F.C Surcharge.
  meter_rent = meter or service rent.
- taxes: one entry per line in the tax / government charges section. Classify by SECTION, not by name.
  gst = sales tax/GST incl. GST on FPA. electricity_duty = ED. municipal_tax = e.g. KE MUCT (KMC).
  other_tax = extra tax, further tax, TV fee, anything else. If only one combined tax amount is printed,
  return taxes: [] and put that amount in total_taxes.
- Do NOT put subtotal lines inside the charges or taxes lists.
- Use null when a value is not printed, blank, covered by a grey box, or unreadable. Never guess.
- Never output names, addresses, CNIC, account/reference/consumer/meter numbers.
- CR always means negative. For any amount printed with CR (e.g. "54770CR"), output it as a STRING exactly as printed, like "54770CR".

Layout guide:
K-Electric (KE) bills, "Bill Calculation" box:
- charges = every line ABOVE the "Electricity Charges" subtotal. total_charges = "Electricity Charges" subtotal.
- taxes = lines between "Electricity Charges" and "Taxes and Duties". total_taxes = "Taxes and Duties" subtotal.
- current_bill = "Electricity Charges for current month".
- units_consumed = the number in the "Current Month ... units" box at the top (e.g. "901 units"), NOT a single meter row.
- reading_date = "Reading Date" printed in the Bill Calculation box (format there is DD-Mon-YY, e.g. 25-Jun-26 = 2026-06-25).
- previous_reading / current_reading: the first row of "Electricity You Have Used".
- arrears = the "Previous Dues" value. NOT "Opening Balance", NOT "Payments/Adjustment".
- payable_within_due_date = "Amount Payable within Due Date" (or "Amount Due").
LESCO and IESCO bills, "BILL CHARGES BREAKDOWN" box:
- charges: "Total Electricity Charges" -> energy; "Subsidies" -> subsidy (negative, skip if blank);
  "Total FPA" or FPA line if printed in that box -> fpa. Any other non-zero charge line in the box -> other.
- "Net Electricity Charges" is a SUBTOTAL: it goes in total_charges, NEVER in the charges list.
- "Taxes" is ONE combined amount: taxes = [] and total_taxes = that amount.
- current_bill = "Current Bill" (normally equals Net Electricity Charges + Taxes; it is NOT the Grand Total).
  payable_within_due_date = "Grand Total" / "Payable within due date" (keep CR).
- arrears = the "Arrears" value (if printed like "39411 / 2", use the first number, 39411; if 0, use 0; if blank, null).
- Values in this box can sit slightly off their labels; match each value to its label row by order:
  Total Electricity Charges, Subsidies, Net Electricity Charges, Taxes, Current Bill, Grand Total.
- Meter readings: if several meter rows are printed, use the FIRST row for previous_reading, current_reading and units_consumed.
- payable_after_due_date: the "After <date>" amount. If it says "NOT TO BE PAID" or is blank, null.

"details" is extra context used later to answer customer questions. Fill what is printed, null otherwise:
{
 "usage_history": [{"month": "YYYY-MM", "units": number, "bill_amount": number or null, "payment": number or null}],
   (every month in the usage/bill history table or chart, oldest first, NOT including the current month unless it is printed in the history),
 "payment_history": [{"month": "YYYY-MM", "billed": number, "paid": number or null, "paid_on": "YYYY-MM-DD" or null}],
 "charge_lines": [{"label": printed label in English, "units": number or null, "rate": number or null, "amount": number}],
 "tax_lines": [{"label": printed label in English, "amount": number}],
 "late_payment_amounts": [{"label": printed label, "amount": number}],
 "printed_percentages": [{"label": string, "percent": number}],
 "connection_info": {"category": string or null, "connection_type": string or null, "meter_rows": [{"previous": number, "present": number, "units": number}], "mdi_kw": number or null, "is_net_metering": true/false},
 "other_printed_amounts": [{"label": string, "amount": number}],
 "notes": short English summary of any messages or notes printed on the bill
}
Output JSON only."""


def _to_number(value):
    """'Rs. 3,430.24' -> 3430.24, '54770CR' -> -54770, '' -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    text = str(value).strip().upper()
    if not text or text in {"NULL", "NONE", "N/A", "-"}:
        return None
    negative = text.endswith("CR") or text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)  # first number, ignores "Rs." etc.
    if not match:
        return None
    num = float(match.group().replace(",", ""))
    num = -num if negative else num
    return int(num) if num.is_integer() else num


def _clean_date(value, pattern):
    if isinstance(value, str) and re.fullmatch(pattern, value.strip()):
        return value.strip()
    return None


def _clean_items(items, allowed, fallback):
    cleaned = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        amount = _to_number(item.get("amount"))
        if amount is None or amount == 0:
            continue
        kind = str(item.get("type", "")).strip().lower()
        cleaned.append({"type": kind if kind in allowed else fallback, "amount": amount})
    return cleaned


def normalize(raw: dict, bill_id: str) -> dict:
    """Force the model output into the exact Level 1 schema."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    prefix = bill_id.split("_")[0].upper()
    out["provider"] = PROVIDER_FROM_FILE.get(prefix) or (raw.get("provider") if raw.get("provider") in {"KE", "LESCO", "IESCO"} else None)
    tariff = raw.get("tariff")
    out["tariff"] = str(tariff).strip() if tariff not in (None, "") else None
    for field in NUMBER_FIELDS:
        out[field] = _to_number(raw.get(field))
    out["bill_month"] = _clean_date(raw.get("bill_month"), r"\d{4}-\d{2}")
    for field in DATE_FIELDS:
        out[field] = _clean_date(raw.get(field), r"\d{4}-\d{2}-\d{2}")

    out["charges"] = _clean_items(raw.get("charges"), CHARGE_TYPES, "other")
    for c in out["charges"]:  # subsidies are always deductions
        if c["type"] == "subsidy" and c["amount"] > 0:
            c["amount"] = -c["amount"]
    out["taxes"] = _clean_items(raw.get("taxes"), TAX_TYPES, "other_tax")

    # LESCO/IESCO print only one combined "Taxes" amount: rule says taxes = [] and amount goes in total_taxes
    if out["provider"] in ("LESCO", "IESCO") and len(out["taxes"]) == 1:
        if out["total_taxes"] is None:
            out["total_taxes"] = out["taxes"][0]["amount"]
        out["taxes"] = []

    # A subtotal must never appear as a charge line
    if out["total_charges"] is not None and len(out["charges"]) > 1:
        out["charges"] = [c for c in out["charges"]
                          if not (c["type"] != "subsidy" and abs(c["amount"] - out["total_charges"]) < 0.01
                                  and abs(sum(x["amount"] for x in out["charges"]) - c["amount"] - out["total_charges"]) < 1)]

    # Rule: if several late-payment amounts are printed, use the highest
    late = [_to_number(v) for v in (raw.get("payable_after_due_date_all") or []) if not isinstance(v, (dict, list))]
    late = [v for v in late + [out["payable_after_due_date"]] if v is not None]
    if late:
        out["payable_after_due_date"] = max(late)

    return {key: out.get(key) for key in FIELD_ORDER}


def empty_result(bill_id: str) -> dict:
    """Used when a bill cannot be read at all: every field present, all null."""
    return normalize({}, bill_id)


LAYOUT_HINTS = """
Extra layout help for this re-read:
- Older LESCO/IESCO layout with three boxes "LESCO CHARGES", "GOVT. CHARGES", "TOTAL CHARGES":
  charges = every non-empty line in LESCO CHARGES except UNITS CONSUMED and TOTAL
  (COST OF ELECTRICITY -> energy, FUEL PRICE ADJUSTMENT -> fpa, F.C SURCHARGE -> surcharge,
  QUARTERLY TARIFF ADJUSTMENT -> quarterly_adjustment, FIXED CHARGES -> fixed, METER RENT / SERVICE RENT -> meter_rent);
  total_charges = the TOTAL of LESCO CHARGES.
  taxes = every non-empty line in GOVT. CHARGES except TOTAL (ELECTRICITY DUTY and ED ON FPA -> electricity_duty,
  GST and GST ON FPA -> gst, INCOME TAX -> income_tax, anything else -> other_tax); total_taxes = the TOTAL of GOVT. CHARGES.
  current_bill = CURRENT BILL in TOTAL CHARGES. arrears = ARREAR/AGE (blank -> null).
- Read every digit carefully, including digits partly covered by pen marks.
- Meter readings must satisfy: present reading - previous reading = units.
"""


def _close(a, b, tol=2):
    return a is not None and b is not None and abs(a - b) <= tol


def _find_mismatches(level1: dict) -> list[str]:
    """Check the bill against itself. Returns a list of problems (empty = consistent)."""
    problems = []
    provider = level1["provider"]
    if provider == "KE":
        for items, total, name in ((level1["charges"], level1["total_charges"], "charges"),
                                   (level1["taxes"], level1["total_taxes"], "taxes")):
            if items and total is not None:
                line_sum = round(sum(i["amount"] for i in items), 2)
                if abs(line_sum - total) > 1:
                    problems.append(f"The {name} lines add up to {line_sum} but the printed {name} subtotal is {total}. "
                                    f"One or more {name} amounts or the subtotal was misread.")
    if provider in ("LESCO", "IESCO"):
        tc, tt, cb = level1["total_charges"], level1["total_taxes"], level1["current_bill"]
        if None not in (tc, tt, cb):
            parts = tc + tt
            pay, arrears = level1["payable_within_due_date"], level1["arrears"] or 0
            if not _close(parts, cb) and not (pay is not None and _close(parts, pay - arrears)):
                problems.append(f"Charges {tc} + taxes {tt} = {round(parts, 2)}, which matches neither the Current Bill "
                                f"{cb} nor the payable amount. One of these values was misread.")
        prev, cur, units = level1["previous_reading"], level1["current_reading"], level1["units_consumed"]
        if None not in (prev, cur, units) and abs((cur - prev) - units) > 1:
            problems.append(f"Present reading {cur} - previous reading {prev} = {round(cur - prev, 2)}, "
                            f"but units consumed is {units}. A reading or the units value was misread.")
    return problems


def extract_bill(image_path: Path) -> tuple[dict, dict]:
    """Return (level1_json, details) for one bill image."""
    bill_id = image_path.stem
    image = load_image(image_path)
    try:
        reply = ask_json(EXTRACT_PROMPT, image=image, cache_key=f"extract_{bill_id}")
    except LLMError as err:
        log.error("Extraction failed for %s: %s", bill_id, err)
        return empty_result(bill_id), {}
    level1 = normalize(reply.get("level1", reply), bill_id)
    details = reply.get("details") if isinstance(reply.get("details"), dict) else {}

    # Self-check: if the printed numbers don't add up, ask the model to re-read once
    problems = _find_mismatches(level1)
    if problems:
        log.warning("%s: self-check failed, re-reading: %s", bill_id, problems)
        fix_prompt = (EXTRACT_PROMPT + "\n\nA previous reading of this bill was inconsistent:\n- "
                    + "\n- ".join(problems)
                    + "\nRe-read those lines digit by digit and return the full corrected JSON.\n" + LAYOUT_HINTS)
        try:
            fixed_reply = ask_json(fix_prompt, image=image, cache_key=f"extract_fix_{bill_id}")
            fixed = normalize(fixed_reply.get("level1", fixed_reply), bill_id)
            if len(_find_mismatches(fixed)) < len(problems):  # keep it only if it really is better
                level1 = fixed
                if isinstance(fixed_reply.get("details"), dict):
                    details = fixed_reply["details"]
                log.info("%s: self-check fixed", bill_id)
        except LLMError as err:
            log.warning("%s: re-read failed, keeping first reading: %s", bill_id, err)
    return level1, details
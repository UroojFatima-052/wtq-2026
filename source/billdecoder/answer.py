"""Level 2: answer customer questions about a bill.

Design: the model is bad at arithmetic, Python is not. So Python computes
every number we can (averages, differences, percentages, history stats),
and the model only picks the right facts and writes the plain-English
answer. All questions for a bill go in ONE call (saves quota, keeps
answers consistent), and the bill image is sent again so the model can
check anything the extraction missed.
"""
import json
import logging
from pathlib import Path

from .llm import LLMError, ask_json, load_image

log = logging.getLogger(__name__)

FAILED_ANSWER = ("Sorry, this answer could not be generated because the bill could not be processed. "
                 "Please check the bill directly.")


def _r(x, digits=2):
    return None if x is None else round(x, digits)


def _pct(part, whole):
    if part is None or not whole:
        return None
    return round(part / whole * 100, 2)


def compute_facts(level1: dict, details: dict) -> dict:
    """Pre-compute numbers in Python so answers are arithmetically correct."""
    facts = {}
    units = level1.get("units_consumed")
    bill = level1.get("current_bill")
    charges = level1.get("charges") or []
    taxes = level1.get("taxes") or []

    # Charges and taxes
    charges_sum = sum(c["amount"] for c in charges)
    taxes_sum = sum(t["amount"] for t in taxes)
    total_charges = level1.get("total_charges") if level1.get("total_charges") is not None else (charges_sum or None)
    total_taxes = level1.get("total_taxes") if level1.get("total_taxes") is not None else (taxes_sum or None)
    facts["sum_of_charge_lines"] = _r(charges_sum)
    facts["sum_of_tax_lines"] = _r(taxes_sum)
    facts["taxes_percent_of_current_bill"] = _pct(total_taxes, bill)
    facts["charges_percent_of_current_bill"] = _pct(total_charges, bill)
    facts["taxes_percent_of_charges"] = _pct(total_taxes, total_charges)
    by_type = {}
    for c in charges:
        by_type[c["type"]] = _r(by_type.get(c["type"], 0) + c["amount"])
    facts["charges_total_by_type"] = by_type
    facts["charge_type_percent_of_current_bill"] = {k: _pct(v, bill) for k, v in by_type.items()}
    tax_by_type = {}
    for t in taxes:
        tax_by_type[t["type"]] = _r(tax_by_type.get(t["type"], 0) + t["amount"])
    facts["taxes_total_by_type"] = tax_by_type
    facts["tax_type_percent_of_current_bill"] = {k: _pct(v, bill) for k, v in tax_by_type.items()}

    # Per-unit costs
    if units:
        facts["average_cost_per_unit_current_bill"] = _r(bill / units) if bill else None
        facts["average_cost_per_unit_before_tax"] = _r(total_charges / units) if total_charges else None
        energy = by_type.get("energy")
        facts["energy_cost_per_unit"] = _r(energy / units) if energy else None

    # Readings check
    prev, cur = level1.get("previous_reading"), level1.get("current_reading")
    if prev is not None and cur is not None:
        facts["reading_difference"] = _r(cur - prev)

    # Late payment
    within, after = level1.get("payable_within_due_date"), level1.get("payable_after_due_date")
    if within is not None and after is not None:
        facts["highest_amount_payable_after_due_date"] = after
        facts["extra_if_paid_after_due_date"] = _r(after - within)
        facts["late_penalty_percent_of_payable"] = _pct(after - within, within)
        facts["late_payment_note"] = ("Late payment answers must state the TOTAL payable after the due date "
                                      "and the extra amount vs paying on time. Surcharge lines are the extra "
                                      "amounts, not totals.")

    # Usage history stats
    history = [h for h in (details.get("usage_history") or [])
            if isinstance(h, dict) and isinstance(h.get("units"), (int, float))]

    # Some bills (e.g. KE's 13-month chart) also show the current month: don't compare it with itself
    month = level1.get("bill_month")
    if history and month and history[-1].get("month") == month:
        history = history[:-1]
        facts["history_note"] = ("The printed usage history also shows the current month; "
                                 "it is excluded from the history averages below.")

    # Net metering: negative units mean more energy exported than imported
    if any(h["units"] < 0 for h in history) or details.get("connection_info", {}).get("is_net_metering"):
        facts["net_metering"] = True
        facts["net_metering_note"] = ("This is a net metering connection. Negative units in the history mean "
                                      "the solar system exported more energy than was imported that month.")
        facts["history_months_with_net_export"] = [f'{h.get("month")} ({h["units"]})' for h in history if h["units"] < 0]
        facts["history_months_with_net_import"] = [f'{h.get("month")} ({h["units"]})' for h in history if h["units"] >= 0]

    # Every late-payment amount printed on the bill (some bills show two stages)
    late = details.get("late_payment_amounts")
    if late:
        facts["all_printed_late_payment_amounts"] = late

    if history:
        vals = [h["units"] for h in history]
        hi = max(history, key=lambda h: h["units"])
        lo = min(history, key=lambda h: h["units"])
        facts["history_months_count"] = len(vals)
        facts["history_months"] = [f'{h.get("month")}: {h["units"]}' for h in history]
        facts["history_average_units"] = _r(sum(vals) / len(vals))
        facts["history_total_units"] = _r(sum(vals))
        facts["history_highest"] = f'{hi.get("month")} ({hi["units"]} units)'
        facts["history_lowest"] = f'{lo.get("month")} ({lo["units"]} units)'
        last3 = vals[-3:]
        facts["history_last_3_months_average_units"] = _r(sum(last3) / len(last3))
        last6 = vals[-6:]
        facts["history_last_6_months_average_units"] = _r(sum(last6) / len(last6))
        if units is not None:
            with_current = vals + [units]
            facts["average_units_including_current_month"] = _r(sum(with_current) / len(with_current))
            avg = sum(vals) / len(vals)
            facts["current_minus_history_average_units"] = _r(units - avg)
            facts["current_vs_history_average_percent_change"] = _pct(units - avg, avg)
            facts["current_minus_previous_month_units"] = _r(units - vals[-1])
            facts["current_vs_previous_month_percent_change"] = _pct(units - vals[-1], vals[-1])
            facts["history_months_above_current"] = sum(1 for v in vals if v > units)

        # Net metering: a net average mixing imports and solar exports is misleading, so replace it
        if any(v < 0 for v in vals):
            imports = [v for v in vals if v >= 0]
            for key in ("history_average_units", "current_minus_history_average_units",
                        "current_vs_history_average_percent_change", "average_units_including_current_month",
                        "history_last_3_months_average_units", "history_last_6_months_average_units"):
                facts.pop(key, None)
            facts["history_months_with_net_import_count"] = len(imports)
            facts["history_months_with_net_export_count"] = len(vals) - len(imports)
            if imports:
                facts["average_units_in_import_months_only"] = _r(sum(imports) / len(imports))
                if units is not None:
                    facts["current_minus_import_months_average"] = _r(units - sum(imports) / len(imports))
        
        for limit in (100, 150, 200, 250, 300, 400, 500):
            above = [f'{h.get("month")} ({h["units"]})' for h in history if h["units"] > limit]
            facts[f"history_months_above_{limit}_units"] = {"count": len(above), "months": above}
        # Same month last year, if present
        if month:
            last_year = f"{int(month[:4]) - 1}{month[4:]}"
            match = [h for h in history if h.get("month") == last_year]
            if match and units is not None:
                facts["same_month_last_year_units"] = match[0]["units"]
                facts["current_minus_same_month_last_year"] = _r(units - match[0]["units"])
    # ---- Seasonal, trend, payment and budgeting facts ----
    hist = [h for h in history if isinstance(h.get("month"), str) and len(h["month"]) == 7]
    hvals = [h["units"] for h in hist]
    if hist:
        summer = [h for h in hist if h["month"][5:7] in ("06", "07", "08")]
        winter = [h for h in hist if h["month"][5:7] in ("12", "01", "02")]
        if summer and winter:
            s_avg = sum(h["units"] for h in summer) / len(summer)
            w_avg = sum(h["units"] for h in winter) / len(winter)
            facts["summer_jun_aug_months"] = [f'{h["month"]}: {h["units"]}' for h in summer]
            facts["winter_dec_feb_months"] = [f'{h["month"]}: {h["units"]}' for h in winter]
            facts["summer_total_units"] = _r(sum(h["units"] for h in summer))
            facts["winter_total_units"] = _r(sum(h["units"] for h in winter))
            facts["summer_average_units"] = _r(s_avg)
            facts["winter_average_units"] = _r(w_avg)
            facts["summer_minus_winter_average_units"] = _r(s_avg - w_avg)
            facts["summer_minus_winter_total_units"] = _r(facts["summer_total_units"] - facts["winter_total_units"])

        if len(hist) >= 3:
            last3 = hist[-3:]
            facts["last_3_history_months"] = [f'{h["month"]}: {h["units"]}' for h in last3]
            first, last = last3[0]["units"], last3[-1]["units"]
            facts["last_3_history_months_direction"] = "up" if last > first else "down" if last < first else "flat"
            if units is not None:
                facts["last_3_months_including_current"] = [f'{h["month"]}: {h["units"]}' for h in hist[-2:]] + \
                                                           [f'{level1.get("bill_month")} (current): {units}']

            # Linear trend forecast for next month (least squares over the history)
            n = len(hvals)
            x_mean, y_mean = (n - 1) / 2, sum(hvals) / n
            denom = sum((i - x_mean) ** 2 for i in range(n))
            slope = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(hvals)) / denom if denom else 0
            next_x = n + (1 if units is not None else 0)  # current month sits after the history
            facts["trend_slope_units_per_month"] = _r(slope)
            facts["next_month_forecast_linear_trend_units"] = _r(y_mean + slope * (next_x - x_mean))
            facts["next_month_forecast_history_average_units"] = _r(y_mean)
            month = level1.get("bill_month")
            if month:
                y, m = int(month[:4]), int(month[5:7])
                nxt = f"{y - 1 if m < 12 else y}-{(m % 12) + 1:02d}"
                same = [h for h in hist if h["month"] == nxt]
                if same:
                    facts["next_month_same_month_last_year_units"] = f'{nxt}: {same[0]["units"]}'
            facts["forecast_note"] = ("Forecasts are estimates: linear trend over the printed history, "
                                      "plus the plain average and the same month last year for comparison.")

        # Payment record (from the history table and the payment history box, whichever is printed)
        record, unpaid = [], []
        for h in hist:
            if h.get("bill_amount") is not None or h.get("payment") is not None:
                record.append(f'{h["month"]}: billed {h.get("bill_amount")}, paid {h.get("payment") if h.get("payment") is not None else "no payment shown"}')
                if not h.get("payment") and (h.get("bill_amount") or 0) > 0:
                    unpaid.append(h["month"])
        for p in details.get("payment_history") or []:
            if isinstance(p, dict):
                record.append(f'{p.get("month")}: billed {p.get("billed")}, paid {p.get("paid")} on {p.get("paid_on")}')
        if record:
            facts["payment_record"] = record
            facts["months_with_bill_but_no_payment_shown"] = unpaid
            facts["payment_note"] = ("A blank payment can mean unpaid or a credit/zero bill; check billed amount. "
                                     "Negative billed amounts are credits.")

        # Budgeting: average monthly bill from printed bill amounts
        bills = [h["bill_amount"] for h in hist if isinstance(h.get("bill_amount"), (int, float))]
        if bills:
            facts["average_monthly_bill_from_history"] = _r(sum(bills) / len(bills))
            facts["average_monthly_bill_months_used"] = len(bills)
        elif units and bill and hvals:
            facts["estimated_average_monthly_bill"] = _r(sum(hvals) / len(hvals) * bill / units)
            facts["estimated_average_monthly_bill_note"] = ("No monthly bill amounts printed for the history; "
                                                            "estimated as average units x this month's cost per unit.")

        # Same units as last month
        if units and bill:
            facts["last_month_units"] = f'{hist[-1]["month"]}: {hist[-1]["units"]}'
            facts["estimated_bill_if_same_units_as_last_month"] = _r(hist[-1]["units"] * bill / units)
            if hist[-1].get("bill_amount") is not None:
                facts["last_month_printed_bill_amount"] = hist[-1]["bill_amount"]

    # Simple cost estimates (clearly labelled as estimates in answers)
    if units and bill:
        rate = bill / units
        facts["estimate_note"] = ("Estimates scale the current bill linearly by units "
                                  "(current_bill / units_consumed per unit). Real slab tariffs are not linear.")
        facts["estimated_bill_for_units"] = {str(u): _r(rate * u) for u in (50, 100, 150, 200, 250, 300, 400, 500)}
        facts["estimated_saving_if_units_reduced_by_10_percent"] = _r(bill * 0.10)
        facts["estimated_saving_if_units_reduced_by_20_percent"] = _r(bill * 0.20)
    return facts


ANSWER_PROMPT = """You are a friendly assistant that explains Pakistani electricity bills to customers.
Answer each customer question about THIS bill (image attached) in plain, clear English.

You have three sources:
1. BILL_JSON: values extracted from the bill.
2. BILL_DETAILS: extra printed details (history, line labels, rates, notes).
3. COMPUTED_FACTS: numbers already calculated exactly in Python. ALWAYS use these numbers for
   averages, differences, counts, percentages and estimates. Do not redo the arithmetic yourself.
If a value in the JSON looks wrong compared with the image, trust what is printed on the image.

Rules for every answer:
- Get numbers right and state them with PKR / units. Be specific: name the lines, months and amounts.
- Use ONLY information on this bill. No outside tariff rates or general knowledge about tariffs.
- If the bill does not show something needed, say clearly that the bill does not show it. Do not guess.
- If you give an estimate, say it is an estimate and briefly explain how it was calculated.
- If a question can be read in more than one way, say which meaning you used, or answer both
  (e.g. with and without the current month).
- Never mention names, addresses, CNIC, account, reference, consumer or meter numbers.
- If the bill shows more than one late-payment stage (e.g. "till 24-Apr" and "after 24-Apr"), mention each amount.
- If COMPUTED_FACTS says net_metering is true, explain that negative units mean net export from solar.
  Do NOT present a negative average as normal usage; describe imports and exports separately instead.
- If the history includes the current month, say whether the comparison includes it.
- Negative payable amounts or "CR" mean a credit: the customer owes nothing and has a balance in their favour.
- Forecasts, budgets and "what would my bill be" answers are ESTIMATES: say so and give the method in a few words.
- For net metering questions, say how you could tell (e.g. net metering text, import/export meter rows, negative units).
- For fuel adjustment, give the amount and the reference month printed next to it (e.g. "FCA Apr-26").
- English only, even if the bill has Urdu. 1 to 4 sentences per answer. No markdown.

BILL_JSON:
{bill_json}

BILL_DETAILS:
{details}

COMPUTED_FACTS:
{facts}

QUESTIONS:
{questions}

Return JSON: {{"answers": [{{"question_id": "Q1", "answer": "..."}}, ...]}} with one entry per question."""


def answer_questions(image_path: Path, level1: dict, details: dict, questions: list[tuple[str, str]]) -> dict:
    """questions: [(question_id, question_text)]. Returns {question_id: answer}."""
    bill_id = image_path.stem
    facts = compute_facts(level1, details)
    prompt = ANSWER_PROMPT.format(
        bill_json=json.dumps(level1, ensure_ascii=False),
        details=json.dumps(details, ensure_ascii=False),
        facts=json.dumps(facts, ensure_ascii=False),
        questions="\n".join(f"{qid}: {q}" for qid, q in questions),
    )
    try:
        reply = ask_json(prompt, image=load_image(image_path), cache_key=f"answers_{bill_id}")
    except LLMError as err:
        log.error("Answering failed for %s: %s", bill_id, err)
        return {qid: FAILED_ANSWER for qid, _ in questions}

    answers = {}
    for item in reply.get("answers", []) if isinstance(reply, dict) else []:
        if isinstance(item, dict) and item.get("question_id") and item.get("answer"):
            answers[str(item["question_id"]).strip()] = " ".join(str(item["answer"]).split())
    missing = [(qid, q) for qid, q in questions if qid not in answers]
    if missing and len(missing) < len(questions):
        # Ask again only for the ones the model skipped
        log.warning("%s: re-asking %d missing answers", bill_id, len(missing))
        answers.update(answer_questions(image_path, level1, details, missing))
    return {qid: answers.get(qid, FAILED_ANSWER) for qid, _ in questions}
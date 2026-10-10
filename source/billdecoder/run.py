"""Command line entry point.

Generate both result CSVs from the templates:
    python -m billdecoder.run --bills <bills folder> --csv <csv folder> --out <output folder>

Only extract (no CSV templates, e.g. on training bills):
    python -m billdecoder.run --bills <bills folder> --out <output folder>
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from .answer import answer_questions
from .extract import empty_result, extract_bill

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
log = logging.getLogger("billdecoder")


def find_image(bills_dir: Path, bill_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        for candidate in (bills_dir / f"{bill_id}{ext}", bills_dir / f"{bill_id}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def process(bills_dir: Path, csv_dir: Path | None, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which bills to process: from level1.csv if given, otherwise every image in the folder
    if csv_dir:
        l1_fields, l1_rows = read_csv(csv_dir / "level1.csv")
        bill_ids = [row["bill_id"].strip() for row in l1_rows]
    else:
        bill_ids = sorted(p.stem for p in bills_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    # Level 1
    results = {}
    for i, bill_id in enumerate(bill_ids, 1):
        image = find_image(bills_dir, bill_id)
        if image is None:
            log.error("[%d/%d] %s: image not found, writing nulls", i, len(bill_ids), bill_id)
            results[bill_id] = (empty_result(bill_id), {}, None)
            continue
        log.info("[%d/%d] Extracting %s", i, len(bill_ids), bill_id)
        level1, details = extract_bill(image)
        results[bill_id] = (level1, details, image)

    (out_dir / "extracted_details.json").write_text(
        json.dumps({b: {"level1": r[0], "details": r[1]} for b, r in results.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    if not csv_dir:
        log.info("Done. Extraction saved to %s", out_dir / "extracted_details.json")
        return

    for row in l1_rows:
        row["json"] = json.dumps(results[row["bill_id"].strip()][0], ensure_ascii=False, separators=(",", ":"))
    write_csv(out_dir / "level1.csv", l1_fields, l1_rows)
    log.info("Wrote %s", out_dir / "level1.csv")

    # Level 2: group questions by bill, one model call per bill
    l2_fields, l2_rows = read_csv(csv_dir / "level2.csv")
    by_bill: dict[str, list[tuple[str, str]]] = {}
    for row in l2_rows:
        by_bill.setdefault(row["bill_id"].strip(), []).append((row["question_id"].strip(), row["question"]))

    all_answers = {}
    for i, (bill_id, questions) in enumerate(by_bill.items(), 1):
        level1, details, image = results.get(bill_id, (empty_result(bill_id), {}, find_image(bills_dir, bill_id)))
        log.info("[%d/%d] Answering %d questions for %s", i, len(by_bill), len(questions), bill_id)
        if image is None:
            all_answers.update({(bill_id, q): "The bill image was not found, so this question cannot be answered."
                                for q, _ in questions})
            continue
        answers = answer_questions(image, level1, details, questions)
        all_answers.update({(bill_id, qid): a for qid, a in answers.items()})

    for row in l2_rows:
        row["answer"] = all_answers[(row["bill_id"].strip(), row["question_id"].strip())]
    write_csv(out_dir / "level2.csv", l2_fields, l2_rows)
    log.info("Wrote %s", out_dir / "level2.csv")

    # Final self-check: no empty cells allowed
    empty1 = sum(1 for r in l1_rows if not r["json"])
    empty2 = sum(1 for r in l2_rows if not r["answer"])
    log.info("Check: level1 rows=%d empty=%d | level2 rows=%d empty=%d", len(l1_rows), empty1, len(l2_rows), empty2)


def main():
    parser = argparse.ArgumentParser(description="Utility Bill Decoder")
    parser.add_argument("--bills", required=True, type=Path, help="Folder with bill images")
    parser.add_argument("--csv", type=Path, help="Folder with level1.csv and level2.csv templates")
    parser.add_argument("--out", default=Path("output"), type=Path, help="Where to write results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.bills.is_dir():
        sys.exit(f"Bills folder not found: {args.bills}")
    if args.csv and not (args.csv / "level1.csv").exists():
        sys.exit(f"level1.csv not found in {args.csv}")
    process(args.bills, args.csv, args.out)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate submission.jsonl for the 30 canonical test pairs in dataset/expanded/test_pairs.json
"""

import json
from pathlib import Path
from bot import compose

EXPANDED_DIR = Path("dataset/expanded")

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_submission():
    test_pairs_data = load_json(EXPANDED_DIR / "test_pairs.json")
    pairs = test_pairs_data.get("pairs", [])

    submission_lines = []

    for item in pairs:
        test_id = item["test_id"]
        trigger_id = item["trigger_id"]
        merchant_id = item["merchant_id"]
        customer_id = item.get("customer_id")

        # Load trigger
        trigger_path = EXPANDED_DIR / "triggers" / f"{trigger_id}.json"
        trigger = load_json(trigger_path)

        # Load merchant
        merchant_path = EXPANDED_DIR / "merchants" / f"{merchant_id}.json"
        merchant = load_json(merchant_path)

        # Load category
        cat_slug = merchant.get("category_slug", "dentists")
        category_path = EXPANDED_DIR / "categories" / f"{cat_slug}.json"
        category = load_json(category_path)

        # Load customer if present
        customer = None
        if customer_id:
            cust_path = EXPANDED_DIR / "customers" / f"{customer_id}.json"
            if cust_path.exists():
                customer = load_json(cust_path)

        # Compose message
        res = compose(category, merchant, trigger, customer)

        submission_line = {
            "test_id": test_id,
            "body": res["body"],
            "cta": res["cta"],
            "send_as": res["send_as"],
            "suppression_key": res["suppression_key"],
            "rationale": res["rationale"]
        }
        submission_lines.append(submission_line)

    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for line in submission_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Successfully generated submission.jsonl with {len(submission_lines)} items.")

if __name__ == "__main__":
    generate_submission()

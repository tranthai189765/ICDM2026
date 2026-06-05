"""
Quick diagnostic: does the FPT (or any) gen LLM obey the [[PRICE: x]] tag?

Sends a handful of representative Buyer and Seller turns with the price-tag
instruction appended (exactly as the live prompts do), prints the RAW reply
plus the parsed tag, and reports a compliance rate.

Usage (on a host with FPT access + .env):
    python test_price_tag.py --model_type fpt --n 5
    python test_price_tag.py --model_type qwen --n 5
"""

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from utils.prompt import call_llm
from utils.generation import (
    PRICE_TAG_INSTRUCTION_BUYER,
    PRICE_TAG_INSTRUCTION_SELLER,
    extract_price_tag,
    strip_price_tag,
)

# A few synthetic single-turn prompts that mirror the live buyer/seller asks.
BUYER_CASES = [
    ("propose 200 (seller asks 500)",
     "You are the Buyer for a used bike listed at $500; your target is $200. "
     "Please propose the price of $200 explicitly. Do not accept any other price. "
     "Please reply with only one short and succinct sentence." + PRICE_TAG_INSTRUCTION_BUYER),
    ("counter 280 (seller asks 500)",
     "You are the Buyer for a used bike listed at $500; your target is $200. "
     "Please counter the seller with the price of $280 explicitly. "
     "Please reply with only one short and succinct sentence." + PRICE_TAG_INSTRUCTION_BUYER),
    ("agree to seller's 350",
     "You are the Buyer. The seller just offered $350. "
     "Please clearly ACCEPT the seller's most recent offered price of $350. "
     "Please reply with only one short and succinct sentence." + PRICE_TAG_INSTRUCTION_BUYER),
    ("inquire, NO price",
     "You are the Buyer for a used bike listed at $500. "
     "Ask the seller a clarifying question about the item's condition. "
     "Do NOT name any price. "
     "Please reply with only one short and succinct sentence." + PRICE_TAG_INSTRUCTION_BUYER),
    ("walk away, NO price",
     "You are the Buyer. Announce you are walking away from this negotiation; "
     "do not propose any new price. "
     "Please reply with only one short and succinct sentence." + PRICE_TAG_INSTRUCTION_BUYER),
]

SELLER_CASES = [
    ("seller holds at 450",
     "You are the Seller of a used bike, asking $500. The buyer offered $200. "
     "Counter by asking $450. Reply with only one short sentence." + PRICE_TAG_INSTRUCTION_SELLER),
    ("seller accepts 350",
     "You are the Seller of a used bike. The buyer offered $350 and you accept. "
     "Reply with only one short sentence." + PRICE_TAG_INSTRUCTION_SELLER),
]


def run_block(title, cases, model_type, n):
    print("\n" + "=" * 80)
    print(f" {title}  (model_type={model_type}, n={n} samples each)")
    print("=" * 80)
    total, ok = 0, 0
    for label, content in cases:
        messages = [{"role": "user", "content": content}]
        replies = call_llm(messages, n=n, temperature=0.0, max_token=64,
                           model_type=model_type)
        print(f"\n--- {label} ---")
        for i, r in enumerate(replies):
            total += 1
            tag = extract_price_tag(r)
            has_tag = "[[PRICE" in (r or "").upper()
            if has_tag:
                ok += 1
            flag = "OK " if has_tag else "MISS"
            print(f"[{flag}] parsed={tag!s:>7} | raw={r!r}")
            print(f"        stripped={strip_price_tag(r)!r}")
    print(f"\n>>> {title}: {ok}/{total} replies contained a [[PRICE: ...]] tag")
    return ok, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="fpt",
                    help="fpt | qwen | llama3 | chatgpt")
    ap.add_argument("--n", type=int, default=3, help="samples per case")
    args = ap.parse_args()

    b_ok, b_tot = run_block("BUYER turns", BUYER_CASES, args.model_type, args.n)
    s_ok, s_tot = run_block("SELLER turns", SELLER_CASES, args.model_type, args.n)

    tot_ok, tot = b_ok + s_ok, b_tot + s_tot
    print("\n" + "#" * 80)
    print(f" OVERALL TAG COMPLIANCE: {tot_ok}/{tot} = {100.0 * tot_ok / max(tot,1):.1f}%")
    print("#" * 80)
    if tot_ok < tot:
        print("Some replies missed the tag. If the rate is low, the tag "
              "instruction prompt may need strengthening before training.")
        sys.exit(0)

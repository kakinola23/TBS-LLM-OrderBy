#!/usr/bin/env python3
"""Amazon price ranking baselines (QS1, pointwise, OrderByLLM merge) on a fixed random sample.

Explicit setting, as in the paper: each product is shown as "<title> — $<price>". Items are keyed
by parent_asin and run through OrderByLLM's passage mode, so the model answers with ids
("A"/"B" or integer ids) instead of copying long titles. Products are listed as "<i>: <text>",
the same rendering TBS uses, and merge uses the TBS listwise prompt unchanged.

  python run_amazon_baselines.py --method merge --n 1000 --m 10 --model GPT-4o
  python run_amazon_baselines.py --method qs1 --n 200 --model GPT-4o
  python run_amazon_baselines.py --method pointwise --n 1000 --model GPT-4o

TBS on the same sample:  python run_amazon.py n1000_gpt4o_m10 1000 --m 10 --model-id gpt-4o --theta 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from loaders.amazon import AMAZON_PRICE_PROMPT_EXPLICIT, explicit_display, load_sample
from methods import merge_sort, pointwise, quicksort
from tbs.llm_clients import cost_usd, make_async_client, make_client
from tbs.metrics import kendalls_tau

SEED = 0

PAIRWISE_PROMPT = """Given two Amazon products with their listed prices:
Product A: {key1}
Product B: {key2}
Which product has the higher price shown? If the prices are equal, the product whose title comes later in alphabetical order counts as higher.
If Product A is higher, select key 'A' as BetterPassageKey. If Product B is higher, select key 'B' as BetterPassageKey."""

POINTWISE_PROMPT = """Given the following Amazon product with its listed price, key: {key}.
Determine the product price shown, in US dollars, as a floating-point number."""


def use_tbs_rendering() -> None:
    """List products as '<i>: <text>' (TBS rendering) instead of Zhao's 'passage_id:<i>' blocks."""

    def create_numbered(passages, usePID=False):
        if usePID:
            return "\n".join(f"{pid}: {p}" for pid, p in passages)
        return "\n".join(f"{i + 1}: {p}" for i, p in enumerate(passages))

    merge_sort.zhao_pc.create_numbered_passages = create_numbered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("tbs", "merge", "qs1", "pointwise"), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--m", type=int, default=10, help="merge batch size")
    ap.add_argument("--model", default="GPT-4o")
    ap.add_argument("--max-inflight", type=int, default=16)
    ap.add_argument("--out-dir", default="results/amazon_baselines")
    args = ap.parse_args()

    sample = load_sample(args.n, SEED)
    asins = list(sample.parent_asin)
    keyed = [(a, explicit_display(t, p)) for a, t, p in zip(sample.parent_asin, sample.display, sample.price)]
    true_order = list(sample.sort_values("gt_rank").parent_asin)
    model_id = make_client(args.model)[1]
    use_tbs_rendering()

    if args.method == "tbs":
        # Default TBS with the same listwise prompt and reply schema as merge (Zhao passage schema).
        import os
        from tbs.passage_sorter import StrictZhaoSorter

        class AmazonExplicitSorter(StrictZhaoSorter):
            def _build_prompt(self, batch, round_id, payload):
                keys = "\n".join(f"{i}: {self.passages[a]}" for i, a in enumerate(batch, 1))
                return AMAZON_PRICE_PROMPT_EXPLICIT.format(keys=keys)

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TBS_CALL_CACHE", str(out / f"n{args.n}_{args.model}_tbs_m{args.m}.calls.jsonl"))
        client, model_id = make_client(args.model)
        text = dict(keyed)
        sorter = AmazonExplicitSorter(
            items=asins, batch_size=args.m, client=client, model=model_id, sort_by="height",
            height_map={a: 0.0 for a in asins}, height_prompt_mode="knowledge", prompt_order="input",
            text_map=text, structured_output=True, query="", passages=text,
            completion_threshold=1.0, enable_completion_early_stop=False, enable_stall_early_stop=False,
            round1_partition="contiguous", round2_mode="recursive", round2_all_rows=False,
            round3_mode="diagonals_then_unresolved", use_transitivity=True, graph_backend="sets")
        t0 = time.time()
        ascending, stats = sorter.sort()  # worst-to-best = cheapest first
        ascending = list(ascending)
        tag = f"n{args.n}_{args.model.replace('.', '_').replace('-', '_')}_tbs_m{args.m}"
        res = {"dataset": "amazon", "prompt": "explicit", "n": args.n, "seed": SEED, "method": "tbs",
               "model": args.model, "model_id": model_id, "m": args.m,
               "tau": round(kendalls_tau(ascending, true_order), 6), "total_calls": stats.count,
               "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
               "cost_usd": round(cost_usd(model_id, stats.prompt_tokens, stats.completion_tokens), 4),
               "round1_calls": sorter.round1_calls, "round2_calls": sorter.round2_calls,
               "complete": sorter.graph.is_complete(), "wall_s": round(time.time() - t0, 1),
               "predicted_order": ascending}
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"amazon {tag}: tau={res['tau']:.4f} calls={res['total_calls']} "
              f"tokens={res['prompt_tokens'] + res['completion_tokens']} cost=${res['cost_usd']}")
        return

    async def run():
        if args.method == "merge":
            client = merge_sort.counting_client(make_async_client(), args.max_inflight)
            best, counts = await merge_sort.merge_sort_passages(
                keyed, AMAZON_PRICE_PROMPT_EXPLICIT, client, model_id, batch_size=args.m)
        elif args.method == "qs1":
            quicksort.install_fallback_counter()
            client = quicksort.limit_concurrency(make_async_client(), args.max_inflight)
            best, counts = await quicksort.quicksort_passages(
                f"amazon_n{args.n}", keyed, PAIRWISE_PROMPT, client, model_id, vote=1)
        else:
            # Item-mode OrderByLLM pointwise (schema key/explanation/value, as on NBA and population);
            # the passage-mode schema names its field relevance_score and the model returns relevances.
            text_to_asin = {t: a for a, t in keyed}
            sync_client = make_client(args.model)[0]
            ascending_text, counts = pointwise.pointwise_items(
                [t for _, t in keyed], POINTWISE_PROMPT, sync_client, model_id, workers=args.max_inflight)
            counts.pop("values", None)
            best = [text_to_asin[t] for t in reversed(ascending_text)]
        return best, counts

    t0 = time.time()
    best_first, counts = asyncio.run(run())
    ascending = list(reversed(best_first))  # best = most expensive, so ascending price = reversed
    if sorted(ascending) != sorted(asins):
        raise SystemExit(f"{args.method}: output does not cover all {args.n} products")
    tag = f"n{args.n}_{args.model.replace('.', '_').replace('-', '_')}_{args.method}" + (
        f"_m{args.m}" if args.method == "merge" else "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"dataset": "amazon", "prompt": "explicit", "n": args.n, "seed": SEED, "method": args.method,
           "model": args.model, "model_id": model_id, "m": args.m if args.method == "merge" else None,
           "tau": round(kendalls_tau(ascending, true_order), 6),
           "total_calls": counts["calls"], "prompt_tokens": counts["prompt_tokens"],
           "completion_tokens": counts["completion_tokens"],
           "cost_usd": round(cost_usd(model_id, counts["prompt_tokens"], counts["completion_tokens"]), 4),
           **{k: v for k, v in counts.items() if k not in ("calls", "prompt_tokens", "completion_tokens")},
           "wall_s": round(time.time() - t0, 1), "predicted_order": ascending}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"amazon {tag}: tau={res['tau']:.4f} calls={res['total_calls']} "
          f"tokens={res['prompt_tokens'] + res['completion_tokens']} cost=${res['cost_usd']}")


if __name__ == "__main__":
    main()

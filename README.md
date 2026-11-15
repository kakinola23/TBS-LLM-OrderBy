# TBS: Efficient Semantic ORDER BY via Transitivity-Aware LLM Batch Sorting

Source code and data for the paper *Efficient Semantic ORDER BY via Transitivity-Aware LLM Batch
Sorting* (Akinkunle Akintayo Akinola, Arijit Khan).

**Full version of the paper (with appendix):** [paper/TBS_full_version.pdf](paper/TBS_full_version.pdf)

**Transitive Batched Sort (TBS)** is a physical operator for `LLM-ORDER BY`. It keeps a record of
every preference the LLM has revealed. It then sends a listwise request only for items whose order
is not already implied, directly or by transitivity. Listwise external merge sort instead re-asks
orderings that earlier calls already settled.

TBS runs in three rounds over a grid of batches of size *m*:

1. sorts the Round-1 batches;
2. stitches the batch heads together (recursive *R0*);
3. sorts diagonals, then packs the still-unresolved pairs until the order is complete (θ = 1) or a
   completion threshold is reached (TBS-EarlyStop, θ < 1).

A reachability index (`tbs/reach_index.py`) answers transitivity queries at the million-item scale.

## Repository layout

```
tbs/                 core library
  sorter.py            TBS Rounds 1-3 (BatchedLLMSorter)
  transitive.py        transitive closure of revealed preferences (sets / chain backends)
  reach_index.py       reachability index (DAG backend, used at scale)
  partition.py         Round-1 membership presets: contiguous, random, similarity, similarity_embed
  passage_sorter.py    TBS listwise adapter for query-conditioned ranking (DL19, SciFact)
  resumable_sorter.py  resumable, parallel TBS for Amazon (with scalable_round3.py, rank_aggregation.py)
  llm_clients.py       OpenAI clients, model table, list prices (USD per 1M tokens)
  call_cache.py        replay cache: every LLM reply is saved; reruns resume for free
  metrics.py           Kendall tau, full-list nDCG, nDCG@10
  batch_size.py        batch-size selection: knee / endpoint utility, call-save rule
  item_runner.py, query_runner.py   shared drivers used by the run_*.py scripts
methods/             one file per method
  tbs.py               TBS and its ablation switches (--no-transitivity, --theta, --round2-*, ...)
  merge_sort.py        OrderByLLM listwise external merge sort (m = 4)
  quicksort.py         OrderByLLM pairwise quicksort, QS1 / QS3
  pointwise.py         OrderByLLM pointwise scoring
loaders/             one loader per dataset (data + dataset-specific prompt adapters)
run_nba.py  run_population.py  run_dl19.py  run_scifact.py  run_movielens.py  run_amazon.py
scripts/             data preparation, batch-size sweep / selection, results summary
data/                datasets (see data/README.md)
third_party/orderbyllm/   OrderByLLM (Zhao et al.), unmodified, MIT, used for the baselines
```

## Datasets

| Dataset | Task | Units × items | Quality |
|---|---|---|---|
| NBA | players by height (model's own knowledge) | 1 × 200 | Kendall τ |
| Population | countries by 2020 population | 1 × 200 | Kendall τ |
| DL19 | passages by relevance to a query | 43 × 100 | full-list nDCG (nDCG_Q) |
| SciFact | abstracts by support for a claim | 50 × 100 | full-list nDCG (nDCG_C) |
| MovieLens-1M | movies by a user's preference | 63 × 70 | full-list nDCG (nDCG_U) |
| Amazon Reviews 2023 | products by price | 1 × 20K / 1M | Kendall τ |

See [`data/README.md`](data/README.md) for sources, licenses and checksums.

## Quick start

### 1. Environment

```bash
conda env create -f environment.yml && conda activate tbs
# or: pip install -r requirements.txt
```

### 2. API key

```bash
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

### 3. Data

NBA, Population, the DL19 run/qrels and the SciFact candidate list ship with the repository. Build
the rest once:

```bash
python scripts/prepare_dl19.py         # MS MARCO passage text for the 43 x 100 DL19 candidates
python scripts/prepare_scifact.py      # BEIR SciFact corpus / queries / qrels
python scripts/prepare_movielens.py    # ML-1M -> 63-user task manifest
# Amazon: download data/amazon/amazon_1m_price_gt.csv (link in data/amazon/README.md) or rebuild it
```

### 4. Run

The five benchmark runners take `--method` and `--model` (`GPT-4o`, `GPT-4.1`, `GPT-4o-mini`). TBS
also takes the batch size `-m`. `run_amazon.py` takes a run name and n; run `--help` for its options.

The defaults are the paper's default TBS configuration: contiguous Round 1, recursive R0 in Round 2,
diagonals then Pack in Round 3, full packing (θ = 1.0), transitive inference on, temperature 0.

```bash
# total orders: one result JSON per run
python run_nba.py        --method tbs --model GPT-4o -m 10 --out results/nba/gpt4o_m10.json
python run_population.py --method tbs --model GPT-4o -m 10 --out results/population/gpt4o_m10.json

# query-conditioned ranking: one JSON line per query/claim/user + a _summary.json
python run_dl19.py      --method tbs --model GPT-4o -m 10
python run_scifact.py   --method tbs --model GPT-4o -m 10
python run_movielens.py --method tbs --model GPT-4o -m 10

# Amazon at scale (GPT-5-nano, explicit prices, m = 40, theta = 0.9)
python run_amazon.py n1m 1000000
```

## Reproducing the paper

| Experiment | Command pattern |
|---|---|
| Baselines vs. TBS | `--method merge`, `--method qs1`, `--method qs3`, `--method pointwise` on any runner (`run_amazon.py` excepted) |
| RQ2 transitivity ablation | add `--no-transitivity` (TBS-NoTrans) |
| RQ2 early stopping | `--theta 0.8 / 0.85 / 0.9 / 0.95` |
| RQ2 Round 2 | `--round2-all-rows` and/or `--round2-mode windows` |
| RQ2 reachability index | `run_amazon.py ... --backend dag` vs `--backend sets` |
| RQ3 models | same command with `--model GPT-4.1 / GPT-4o / GPT-4o-mini`, each at its own `-m`. GPT-4o-mini on NBA uses `--structured-schema recall` |
| RQ4 batch size | `python scripts/sweep_batch_size.py --dataset movielens --model GPT-4o`, then `python scripts/select_batch_size.py results/sweep/movielens/GPT_4o --rule call-save` |
| RQ5 Round-1 membership | `--round1-partition random / similarity / similarity_embed` |
| Pointwise vs. explicit values | `run_nba.py --method pointwise --out pw.json`, then `run_nba.py --method tbs -m 10 --values-from pw.json` |

`python scripts/summarize_results.py` collects every result into `results/summary.csv`: quality,
calls, prompt/completion tokens and USD cost at list prices.

## Outputs, resuming and cost

- **Saved per run:** the full predicted order, the quality metric, API calls, prompt and completion
  tokens, and the cost in USD.
- **Resuming:** every LLM reply is appended to a `*.calls.jsonl` replay cache next to the result. If
  a run stops for any reason, rerun the same command. Finished calls are replayed, with no API cost,
  and the call and token totals equal those of an uninterrupted run.
- **Model omissions:** no item is ever dropped. TBS re-asks a structured listwise batch whose reply
  omits items, up to 5 attempts; anything still missing keeps its input position. The merge and
  pointwise baselines re-ask any batch or value whose OrderByLLM retries were exhausted.

## Citation

```bibtex
@inproceedings{akinola2027tbs,
  title     = {Efficient Semantic ORDER BY via Transitivity-Aware LLM Batch Sorting},
  author    = {Akinola, Akinkunle Akintayo and Khan, Arijit},
  year      = {2027}
}
```

## Acknowledgment

- **Baselines and prompts:** OrderByLLM (Zhao et al.), https://github.com/ZhaoFuheng/OrderByLLM.
- **Datasets:** TREC DL19 / MS MARCO, BEIR SciFact, MovieLens-1M (GroupLens), and Amazon Reviews 2023
  (McAuley Lab).

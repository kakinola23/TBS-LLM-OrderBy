"""MovieLens-1M preference ranking (Phase-1 slice: 63 users x 70 candidate movies).

Each user has a short rating-history profile and 70 candidate movies; ground truth orders the
candidates by the user's held-out rating. ``data/movielens/movielens_phase1_manifest.json`` holds
the sampled tasks (build it from ML-1M with ``python scripts/prepare_movielens.py``).

Quality metric nDCG_U: full-list nDCG with exponential gain (2^rel - 1), ideal order by relevance.
TBS uses a plain-text MovieLens listwise prompt (comma-separated ids, no structured output).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tbs.metrics import kendalls_tau
from tbs.retry import robust_chat_completion
from tbs.sorter import BatchedLLMSorter

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "movielens" / "movielens_phase1_manifest.json"

ML_SYSTEM = (
    "You are a movie recommendation oracle. Given a user's viewing history and a list "
    "of candidate movies, return the movie IDs in order from most preferred to least "
    "preferred for this user, separated by commas. Output only the IDs, no explanation."
)


class MovieLensPackingSorter(BatchedLLMSorter):
    """BatchedLLMSorter with MovieLens preference prompts + quiet extraction."""

    def __init__(
        self,
        *args: Any,
        profile_text: str,
        display_map: Dict[str, str],
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("sort_by", "height")
        kwargs.setdefault("height_map", {})
        kwargs.setdefault("height_prompt_mode", "explicit")
        super().__init__(*args, **kwargs)
        self.profile_text = profile_text
        self.display_map = display_map

    def _build_prompt(self, batch: Sequence[str], round_id: int, payload: Dict[str, object]) -> str:
        lines = [
            self.profile_text,
            "",
            "Candidate movies (movie ID followed by title and genres). "
            "Rank them by how much this user would prefer them (most preferred first).",
            "",
        ]
        for mid in batch:
            lines.append(f"[{mid}] {self.display_map.get(mid, mid)}")
        lines.extend(
            [
                "",
                "Return ONLY the movie IDs in order from most preferred to least preferred, "
                "separated by commas. No explanation.",
            ]
        )
        return "\n".join(lines)

    def _sort_with_llm_once(
        self,
        presentation: Sequence[str],
        round_id: int,
        payload: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        """One listwise call, re-asked up to five attempts until every movie id is listed;
        ids still missing after that are appended in presentation order."""
        prompt = self._build_prompt(presentation, round_id, payload or {})
        n = len(presentation)
        best: List[str] = []
        for attempt in range(1, 6):
            suffix = "" if attempt == 1 else (
                f"\n\nYour previous answer was incomplete. List all {n} movie IDs exactly once: "
                + ", ".join(presentation)
            )
            response = robust_chat_completion(
                self.client,
                self.model,
                ML_SYSTEM,
                prompt + suffix,
                max_retries=getattr(self, "llm_max_retries", 6),
            )
            text = (getattr(response.choices[0].message, "content", "") or "").strip()
            self.stats.update(list(presentation), response)
            ordered = self._extract_movie_ids(text, presentation)
            if len(ordered) == n:
                return ordered
            if len(ordered) > len(best):
                best = ordered
        return best + [m for m in presentation if m not in best]

    def _parse_movie_ids(self, text: str, batch: Sequence[str]) -> List[str]:
        ordered = self._extract_movie_ids(text, batch)
        return ordered + [m for m in batch if m not in ordered]

    def _extract_movie_ids(self, text: str, batch: Sequence[str]) -> List[str]:
        """Movie ids in the order the reply lists them (first occurrence only)."""
        batch_set = set(batch)
        ordered: List[str] = []
        tokens = text.replace("\n", ",").split(",")
        for tok in tokens:
            clean = tok.strip().strip("[]`\"' ")
            if clean in batch_set and clean not in ordered:
                ordered.append(clean)
        if len(ordered) < len(batch):
            # substring fallback for glued responses
            remaining = [m for m in batch if m not in ordered]
            pos = []
            for m in remaining:
                i = text.find(m)
                if i >= 0:
                    pos.append((i, m))
            pos.sort()
            for _, m in pos:
                if m not in ordered:
                    ordered.append(m)
        return ordered

    def sort(self):
        # Same schedule as BatchedLLMSorter.sort, with the quiet linear extension.
        calls_before = self.stats.count
        columns = self._round_one()
        self.round1_calls = self.stats.count - calls_before
        if self.graph.is_complete():
            self.round2_calls = 0
            return self._quiet_total_order(), self.stats
        calls_before_r2 = self.stats.count
        self._round_two(columns)
        self.round2_calls = self.stats.count - calls_before_r2
        if self.graph.is_complete():
            return self._quiet_total_order(), self.stats
        self._round_three(columns)
        return self._quiet_total_order(), self.stats



Task = Tuple[str, str, List[str], List[str], Dict[str, int], Dict[str, str]]


def load_tasks(manifest_path: Path = MANIFEST) -> List[Task]:
    """[(user_id, profile_text, movie_ids in input order, ground-truth order, relevance, display)]."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [
        (str(t["user_id"]), t["profile_text"], t["movie_ids"], t["ground_truth_order"],
         {k: int(v) for k, v in t["relevance_map"].items()}, t["display_map"])
        for t in manifest["tasks"]
    ]


def dcg_at_k(order: List[str], relevance_map: Dict[str, int], k: Optional[int] = None) -> float:
    if k is not None:
        order = order[:k]
    return sum((math.pow(2, relevance_map.get(d, 0)) - 1.0) / math.log2(i + 2) for i, d in enumerate(order))


def ndcg_at_k(predicted_order: List[str], ground_truth_order: List[str], relevance_map: Dict[str, int],
              k: Optional[int] = None) -> float:
    """nDCG@k (k=None: full list = nDCG_U). Ideal order: relevance desc, ties by ground-truth position."""
    gt_pos = {d: i for i, d in enumerate(ground_truth_order)}
    ideal = sorted(ground_truth_order, key=lambda d: (-relevance_map.get(d, 0), gt_pos[d]))
    idcg = dcg_at_k(ideal, relevance_map, k)
    if idcg <= 0:
        return 1.0
    return dcg_at_k(predicted_order, relevance_map, k) / idcg


def evaluate(predicted_order: List[str], ground_truth_order: List[str], relevance_map: Dict[str, int]) -> Dict[str, float]:
    return {
        "ndcg": round(ndcg_at_k(predicted_order, ground_truth_order, relevance_map, k=None), 6),
        "ndcg10": round(ndcg_at_k(predicted_order, ground_truth_order, relevance_map, k=10), 6),
        "tau": round(kendalls_tau(predicted_order, ground_truth_order), 6),
    }


class MovieLensPairwiseComparator:
    """Pairwise preference oracle for quicksort; the two movies are shown in a seeded random order."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        profile_text: str,
        display_map: Dict[str, str],
        rng: random.Random,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.profile_text = profile_text
        self.display_map = display_map
        self.rng = rng
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.fallbacks = 0

    def _system_message(self) -> str:
        return (
            "You are a movie recommendation oracle. Given a user's viewing history and two "
            "candidate movies, return the movie ID the user would prefer more, as a single ID. "
            "Output only the ID, no explanation."
        )

    def _build_prompt(self, movie_a: str, movie_b: str) -> str:
        lines = [
            self.profile_text,
            "",
            "Candidate movies (movie ID followed by title and genres). "
            "Which would this user prefer more?",
            "",
            f"[{movie_a}] {self.display_map.get(movie_a, movie_a)}",
            f"[{movie_b}] {self.display_map.get(movie_b, movie_b)}",
            "",
            "Return ONLY the preferred movie ID. No explanation.",
        ]
        return "\n".join(lines)

    async def compare(self, movie_a: str, movie_b: str) -> int:
        if movie_a == movie_b:
            return 0

        presentation = [movie_a, movie_b]
        self.rng.shuffle(presentation)

        prompt = self._build_prompt(presentation[0], presentation[1])
        response = robust_chat_completion(
            self.client,
            self.model_id,
            self._system_message(),
            prompt,
        )
        self.calls += 1
        usage = getattr(response, "usage", None)
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        text = (getattr(response.choices[0].message, "content", "") or "").strip()
        winner = None
        for mid in presentation:
            if mid in text:
                winner = mid
                break
        if winner is None:
            for token in text.replace("\n", " ").split():
                clean = token.strip("[]`,.")
                if clean in presentation:
                    winner = clean
                    break
        if winner is None:
            self.fallbacks += 1
            winner = presentation[0]

        return -1 if winner == movie_a else 1



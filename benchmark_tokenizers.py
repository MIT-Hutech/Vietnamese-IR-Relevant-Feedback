"""Compare the original regex tokenizer with Underthesea on the IRS dataset."""

from __future__ import annotations

import copy
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import ir_engine as engine


DATA_PATH = Path(__file__).parent / "data" / "dataset.json"
TOP_K = 20
MIN_JUDGED_DOCS = 5


def regex_tokenize(text: str) -> list[str]:
    """Tokenizer used by the project before switching to Underthesea."""
    tokens = re.findall(
        r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩị"
        r"óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+",
        text.lower(),
    )
    return [t for t in tokens if t not in engine.VI_STOPWORDS and len(t) > 1]


def title_key(record: dict) -> str:
    return record.get("title", "").strip().lower()


def build_qrels(records: list[dict]) -> dict[str, dict[str, int]]:
    """Create weak qrels from the dataset's query and relevance columns."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        query = record.get("query", "").strip()
        key = title_key(record)
        if not query or not key:
            continue
        grade = int(record.get("relevance", 1))
        qrels[query][key] = max(grade, qrels[query].get(key, 0))
    return {
        query: judgments
        for query, judgments in qrels.items()
        if len(judgments) >= MIN_JUDGED_DOCS
        and any(grade >= 2 for grade in judgments.values())
    }


def metrics(results: list[dict], judgments: dict[str, int]) -> dict[str, float]:
    relevant = {key for key, grade in judgments.items() if grade >= 2}
    ranked_keys = [title_key(result["doc"]) for result in results]

    hits_at_10 = sum(key in relevant for key in ranked_keys[:10])
    precision_10 = hits_at_10 / 10
    recall_10 = hits_at_10 / len(relevant) if relevant else 0.0

    hits = 0
    ap_sum = 0.0
    for rank, key in enumerate(ranked_keys[:TOP_K], 1):
        if key in relevant:
            hits += 1
            ap_sum += hits / rank
    ap_20 = ap_sum / len(relevant) if relevant else 0.0

    dcg = 0.0
    for rank, key in enumerate(ranked_keys[:10], 1):
        grade = judgments.get(key, 1)
        gain = (2 ** (grade - 1) - 1) if grade >= 2 else 0
        dcg += gain / math.log2(rank + 1)
    ideal_gains = sorted(
        ((2 ** (grade - 1) - 1) for grade in judgments.values() if grade >= 2),
        reverse=True,
    )[:10]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    ndcg_10 = dcg / idcg if idcg else 0.0

    return {
        "P@10": precision_10,
        "Recall@10": recall_10,
        "AP@20": ap_20,
        "nDCG@10": ndcg_10,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def evaluate(
    name: str,
    tokenizer,
    records: list[dict],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict, dict[str, dict]]:
    original_tokenizer = engine._tokenize
    engine._tokenize = tokenizer
    try:
        index = engine.IRIndex()
        started = time.perf_counter()
        index.build(copy.deepcopy(records))
        build_seconds = time.perf_counter() - started

        per_query: dict[str, dict] = {}
        search_times = []
        initial_rows = []
        rocchio_rows = []
        rf_queries = 0

        for query, title_judgments in sorted(qrels.items()):
            started = time.perf_counter()
            initial_results = index.search(query, top_k=TOP_K)
            search_times.append(time.perf_counter() - started)
            initial = metrics(initial_results, title_judgments)
            initial_rows.append(initial)

            relevant_ids = {
                result["doc_id"]
                for result in initial_results
                if title_judgments.get(title_key(result["doc"]), 1) >= 2
            }
            non_relevant_ids = {
                result["doc_id"]
                for result in initial_results
                if title_judgments.get(title_key(result["doc"])) == 1
            }

            after = initial
            if relevant_ids:
                rf_queries += 1
                query_vec = index.query_to_vec(query)
                new_query_vec = engine.rocchio(
                    query_vec,
                    [index.tfidf_vecs[i] for i in relevant_ids],
                    [index.tfidf_vecs[i] for i in non_relevant_ids],
                    alpha=1.0,
                    beta=0.75,
                    gamma=0.15,
                )
                after_results = index.search_by_vec(new_query_vec, top_k=TOP_K)
                after = metrics(after_results, title_judgments)
            rocchio_rows.append(after)
            per_query[query] = {"initial": initial, "after_rocchio": after}

        summary = {
            "name": name,
            "documents": len(index.docs),
            "vocabulary": len(index.vocab),
            "average_document_length": index.avg_dl,
            "build_seconds": build_seconds,
            "mean_query_ms": statistics.fmean(search_times) * 1000,
            "p95_query_ms": sorted(search_times)[max(0, math.ceil(len(search_times) * 0.95) - 1)] * 1000,
            "evaluated_queries": len(qrels),
            "rocchio_queries": rf_queries,
            "initial": mean_metrics(initial_rows),
            "after_rocchio": mean_metrics(rocchio_rows),
        }
        return summary, per_query
    finally:
        engine._tokenize = original_tokenizer


def print_markdown(summaries: list[dict]) -> None:
    print("\n## Retrieval quality")
    print("| Tokenizer | Stage | P@10 | Recall@10 | MAP@20 | nDCG@10 |")
    print("|---|---:|---:|---:|---:|---:|")
    for summary in summaries:
        for stage, label in (("initial", "Initial"), ("after_rocchio", "After Rocchio")):
            row = summary[stage]
            print(
                f"| {summary['name']} | {label} | {row['P@10']:.4f} | "
                f"{row['Recall@10']:.4f} | {row['AP@20']:.4f} | {row['nDCG@10']:.4f} |"
            )

    print("\n## Runtime and index")
    print("| Tokenizer | Documents | Vocabulary | Avg. doc length | Build (s) | Mean query (ms) | P95 query (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        print(
            f"| {summary['name']} | {summary['documents']} | {summary['vocabulary']} | "
            f"{summary['average_document_length']:.2f} | {summary['build_seconds']:.3f} | "
            f"{summary['mean_query_ms']:.3f} | {summary['p95_query_ms']:.3f} |"
        )


def main() -> None:
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    qrels = build_qrels(records)
    underthesea_tokenizer = engine._tokenize

    summaries = []
    details = {}
    for name, tokenizer in (
        ("Regex", regex_tokenize),
        ("Underthesea", underthesea_tokenizer),
    ):
        summary, per_query = evaluate(name, tokenizer, records, qrels)
        summaries.append(summary)
        details[name] = per_query

    print(f"Evaluated queries: {len(qrels)}; top-k: {TOP_K}; relevant grades: 2-3")
    print_markdown(summaries)

    deltas = []
    for query in qrels:
        regex_ap = details["Regex"][query]["initial"]["AP@20"]
        uts_ap = details["Underthesea"][query]["initial"]["AP@20"]
        deltas.append((uts_ap - regex_ap, query, regex_ap, uts_ap))
    print("\n## Largest initial AP@20 changes")
    for delta, query, before, after in sorted(deltas, reverse=True)[:5]:
        print(f"GAIN {delta:+.4f}: {query!r} ({before:.4f} -> {after:.4f})")
    for delta, query, before, after in sorted(deltas)[:5]:
        print(f"LOSS {delta:+.4f}: {query!r} ({before:.4f} -> {after:.4f})")


if __name__ == "__main__":
    main()



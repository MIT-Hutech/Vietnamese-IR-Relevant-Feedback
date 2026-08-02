"""
Vietnamese IR Engine:
- BM25 retrieval
- TF-IDF vectorization (from scratch, no sklearn dep)
- Rocchio Algorithm for Relevant Feedback
- Metrics: Precision@K, MAP, nDCG@K
"""

import json
import math
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ── Stopwords tiếng Việt (basic) ────────────────────────────────────────────
VI_STOPWORDS = {
    "và","của","là","có","được","trong","cho","với","các","những",
    "này","đó","một","như","tại","từ","khi","về","theo","hay",
    "thì","mà","đã","sẽ","bị","bởi","vì","nên","nhưng","hoặc",
    "để","cũng","vậy","sau","trước","đây","cần","không","phải",
    "nhiều","rất","hơn","nhất","lại","đến","trên","dưới","giữa",
    "mọi","do","tuy","dù","vẫn","đang","qua","nào","ai","gì",
    "thế","còn","thêm","hiện","nay","năm","ngày","người","tháng",
    "the","a","an","is","are","was","were","in","on","at","of",
    "to","for","and","or","but","with","by","from","as"
}


def _tokenize(text: str) -> list[str]:
    """Basic Vietnamese tokenizer: lowercase + split on non-alpha."""
    text = text.lower()
    # Keep Vietnamese characters + latin + digits
    tokens = re.findall(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+', text)
    return [t for t in tokens if t not in VI_STOPWORDS and len(t) > 1]


# ── Index ────────────────────────────────────────────────────────────────────

class IRIndex:
    """Inverted index + BM25 + TF-IDF vectors."""

    def __init__(self):
        self.docs: list[dict] = []          # raw records
        self.doc_tokens: list[list[str]] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term → [(doc_id, freq)]
        self.df: dict[str, int] = {}        # document frequency
        self.doc_len: list[int] = []
        self.avg_dl: float = 0.0
        self.vocab: set[str] = set()
        self.vocab_list: list[str] = []
        self.tfidf_vecs: list[dict[str, float]] = []  # sparse TF-IDF vectors
        self.built = False

        # BM25 params
        self.k1 = 1.5
        self.b  = 0.75

    def build(self, records: list[dict]):
        """Index all records, deduplicating by title."""
        seen: set[str] = set()
        unique: list[dict] = []
        for r in records:
            key = r.get("title", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        # Re-assign sequential IDs
        for i, r in enumerate(unique):
            r["id"] = i + 1
        self.docs = unique
        N = len(unique)

        # Tokenize
        for rec in unique:
            text = f"{rec.get('title','')} {rec.get('query','')} {rec.get('document','')}"
            tokens = _tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

        self.avg_dl = sum(self.doc_len) / max(N, 1)

        # Build inverted index + DF
        for doc_id, tokens in enumerate(self.doc_tokens):
            freq = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            for t, f in freq.items():
                self.inverted[t].append((doc_id, f))
                self.df[t] = self.df.get(t, 0) + 1
                self.vocab.add(t)

        self.vocab_list = sorted(self.vocab)

        # Build TF-IDF vectors (sparse)
        for doc_id, tokens in enumerate(self.doc_tokens):
            freq = defaultdict(int)
            for t in tokens: freq[t] += 1
            vec = {}
            dl = self.doc_len[doc_id]
            for t, f in freq.items():
                tf = f / max(dl, 1)
                idf = math.log((N + 1) / (self.df.get(t, 0) + 1)) + 1
                vec[t] = tf * idf
            # Normalize
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.tfidf_vecs.append({t: v/norm for t, v in vec.items()})

        self.built = True

    def _idf(self, term: str) -> float:
        N = len(self.docs)
        df = self.df.get(term, 0)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, doc_id: int, query_terms: list[str]) -> float:
        score = 0.0
        dl = self.doc_len[doc_id]
        freq_map = defaultdict(int)
        for t in self.doc_tokens[doc_id]: freq_map[t] += 1
        for term in query_terms:
            f = freq_map.get(term, 0)
            idf = self._idf(term)
            tf_norm = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1)))
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 search. Returns list of {doc, score, rank}."""
        if not self.built:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        # Candidate docs from inverted index
        candidates: set[int] = set()
        for t in q_terms:
            for doc_id, _ in self.inverted.get(t, []):
                candidates.add(doc_id)

        # Score
        scored = []
        for doc_id in candidates:
            score = self.bm25_score(doc_id, q_terms)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: -x[1])

        results = []
        for rank, (doc_id, score) in enumerate(scored[:top_k], 1):
            results.append({
                "doc": self.docs[doc_id],
                "doc_id": doc_id,
                "score": round(score, 4),
                "rank": rank,
            })
        return results

    def query_to_vec(self, query: str) -> dict[str, float]:
        """Convert query string to TF-IDF-like unit vector."""
        tokens = _tokenize(query)
        if not tokens: return {}
        freq = defaultdict(int)
        for t in tokens: freq[t] += 1
        N = len(self.docs)
        vec = {}
        for t, f in freq.items():
            tf = f / len(tokens)
            idf = math.log((N + 1) / (self.df.get(t, 0) + 1)) + 1
            vec[t] = tf * idf
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t: v/norm for t, v in vec.items()}

    def vec_to_query_str(self, vec: dict[str, float], top_n: int = 10) -> str:
        """Get top-N terms from a vector as a query string."""
        sorted_terms = sorted(vec.items(), key=lambda x: -x[1])
        return " ".join(t for t, _ in sorted_terms[:top_n])

    def search_by_vec(self, query_vec: dict[str, float], top_k: int = 20) -> list[dict]:
        """Cosine similarity search using a query vector (after Rocchio)."""
        if not self.built or not query_vec: return []

        scores: dict[int, float] = defaultdict(float)
        q_norm = math.sqrt(sum(v*v for v in query_vec.values())) or 1.0

        for term, q_weight in query_vec.items():
            if q_weight <= 0: continue
            for doc_id, _ in self.inverted.get(term, []):
                d_weight = self.tfidf_vecs[doc_id].get(term, 0)
                scores[doc_id] += (q_weight / q_norm) * d_weight

        scored = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for rank, (doc_id, score) in enumerate(scored[:top_k], 1):
            results.append({
                "doc": self.docs[doc_id],
                "doc_id": doc_id,
                "score": round(score, 4),
                "rank": rank,
            })
        return results


# ── Rocchio Algorithm ────────────────────────────────────────────────────────

def rocchio(
    query_vec: dict[str, float],
    relevant_vecs: list[dict[str, float]],
    non_relevant_vecs: list[dict[str, float]],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> dict[str, float]:
    """
    Rocchio formula:
    q_m = α*q_0 + β*(1/|R|)*Σ(r∈R) d_r - γ*(1/|NR|)*Σ(nr∈NR) d_nr
    """
    new_vec: dict[str, float] = {}

    # α * q_0
    for t, w in query_vec.items():
        new_vec[t] = new_vec.get(t, 0) + alpha * w

    # β * (1/|R|) * Σ d_r
    if relevant_vecs:
        n_r = len(relevant_vecs)
        for vec in relevant_vecs:
            for t, w in vec.items():
                new_vec[t] = new_vec.get(t, 0) + (beta / n_r) * w

    # - γ * (1/|NR|) * Σ d_nr
    if non_relevant_vecs:
        n_nr = len(non_relevant_vecs)
        for vec in non_relevant_vecs:
            for t, w in vec.items():
                new_vec[t] = new_vec.get(t, 0) - (gamma / n_nr) * w

    # Remove zero/negative weights (keep only positive)
    new_vec = {t: w for t, w in new_vec.items() if w > 0}

    # Normalize
    norm = math.sqrt(sum(v*v for v in new_vec.values())) or 1.0
    return {t: v/norm for t, v in new_vec.items()}


def pseudo_rf(
    query_vec: dict[str, float],
    top_results: list[dict],
    index: IRIndex,
    top_k: int = 5,
    beta: float = 0.5,
) -> dict[str, float]:
    """Pseudo Relevance Feedback: treat top-K as relevant."""
    relevant_vecs = [index.tfidf_vecs[r["doc_id"]] for r in top_results[:top_k]]
    return rocchio(query_vec, relevant_vecs, [], alpha=1.0, beta=beta, gamma=0.0)


# ── Metrics ──────────────────────────────────────────────────────────────────

def precision_at_k(results: list[dict], relevant_ids: set[int], k: int) -> float:
    top = results[:k]
    hits = sum(1 for r in top if r["doc_id"] in relevant_ids)
    return hits / k if k > 0 else 0.0


def average_precision(results: list[dict], relevant_ids: set[int]) -> float:
    if not relevant_ids: return 0.0
    hit = 0; ap = 0.0
    for i, r in enumerate(results, 1):
        if r["doc_id"] in relevant_ids:
            hit += 1
            ap += hit / i
    return ap / len(relevant_ids)


def ndcg_at_k(results: list[dict], relevant_ids: set[int], k: int) -> float:
    dcg = sum(
        (1 / math.log2(i + 2))
        for i, r in enumerate(results[:k])
        if r["doc_id"] in relevant_ids
    )
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(results: list[dict], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids: return 0.0
    top = results[:k]
    hits = sum(1 for r in top if r["doc_id"] in relevant_ids)
    return hits / len(relevant_ids)


def compute_metrics(results: list[dict], relevant_ids: set[int]) -> dict:
    return {
        "P@5":   round(precision_at_k(results, relevant_ids, 5), 4),
        "P@10":  round(precision_at_k(results, relevant_ids, 10), 4),
        "R@10":  round(recall_at_k(results, relevant_ids, 10), 4),
        "AP":    round(average_precision(results, relevant_ids), 4),
        "nDCG@5": round(ndcg_at_k(results, relevant_ids, 5), 4),
        "nDCG@10":round(ndcg_at_k(results, relevant_ids, 10), 4),
    }


# ── Singleton index ──────────────────────────────────────────────────────────

_index: Optional[IRIndex] = None


def get_index(data_path: str) -> IRIndex:
    global _index
    if _index is None:
        _index = IRIndex()
        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)
        _index.build(records)
    return _index

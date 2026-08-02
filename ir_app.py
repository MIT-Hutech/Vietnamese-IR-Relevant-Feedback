import json
import uuid
import logging
from pathlib import Path
from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS

import ir_engine as engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "ir-rf-demo-2024"
CORS(app)

DATA_PATH = str(Path(__file__).parent / "data" / "dataset.json")

# In-memory session store (keyed by session_id)
# Each session: { query, query_vec, results, feedback, rounds, metrics_history }
_sessions: dict[str, dict] = {}

TOP_K = 20   # results per query
RF_PARAMS = {"alpha": 1.0, "beta": 0.75, "gamma": 0.15}


def _get_session(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {
            "query": "",
            "query_vec": {},
            "original_query_vec": {},
            "results": [],
            "feedback": {},     # doc_id (str) → "R" | "NR"
            "rounds": [],       # list of round summaries
            "metrics_history": [],
        }
    return _sessions[sid]


@app.route("/")
def index():
    idx = engine.get_index(DATA_PATH)
    stats = {
        "total_docs": len(idx.docs),
        "vocab_size": len(idx.vocab),
        "avg_dl": round(idx.avg_dl, 1),
    }
    return render_template("ir_index.html", stats=stats)


@app.route("/api/search", methods=["POST"])
def api_search():
    body = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    sid = body.get("session_id") or str(uuid.uuid4())

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    idx = engine.get_index(DATA_PATH)
    sess = _get_session(sid)

    # Fresh search
    results = idx.search(query, top_k=TOP_K)
    query_vec = idx.query_to_vec(query)

    sess.update({
        "query": query,
        "query_vec": query_vec,
        "original_query_vec": dict(query_vec),
        "results": results,
        "feedback": {},
        "rounds": [],
        "metrics_history": [],
    })

    # Top query terms for display
    top_terms = sorted(query_vec.items(), key=lambda x: -x[1])[:8]

    return jsonify({
        "session_id": sid,
        "query": query,
        "total_results": len(results),
        "results": _format_results(results, {}),
        "query_terms": [{"term": t, "weight": round(w, 4)} for t, w in top_terms],
        "round": 0,
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Apply Rocchio based on user feedback."""
    body = request.get_json(silent=True) or {}
    sid = body.get("session_id", "")
    feedback = body.get("feedback", {})   # {str(doc_id): "R" | "NR"}
    alpha = float(body.get("alpha", RF_PARAMS["alpha"]))
    beta  = float(body.get("beta",  RF_PARAMS["beta"]))
    gamma = float(body.get("gamma", RF_PARAMS["gamma"]))

    if sid not in _sessions:
        return jsonify({"error": "Session not found"}), 404

    sess = _sessions[sid]
    idx = engine.get_index(DATA_PATH)

    # Save feedback
    sess["feedback"].update(feedback)

    relevant_ids = {int(k) for k, v in sess["feedback"].items() if v == "R"}
    non_rel_ids  = {int(k) for k, v in sess["feedback"].items() if v == "NR"}

    relevant_vecs     = [idx.tfidf_vecs[i] for i in relevant_ids  if i < len(idx.tfidf_vecs)]
    non_relevant_vecs = [idx.tfidf_vecs[i] for i in non_rel_ids   if i < len(idx.tfidf_vecs)]

    if not relevant_vecs:
        return jsonify({"error": "Cần đánh dấu ít nhất 1 tài liệu là Liên quan (R)"}), 400

    old_results = sess["results"]

    # Rocchio
    new_query_vec = engine.rocchio(
        sess["query_vec"], relevant_vecs, non_relevant_vecs,
        alpha=alpha, beta=beta, gamma=gamma
    )

    # New results
    new_results = idx.search_by_vec(new_query_vec, top_k=TOP_K)

    # Metrics (treating user-marked relevant docs as ground truth)
    metrics_before = engine.compute_metrics(old_results, relevant_ids)
    metrics_after  = engine.compute_metrics(new_results, relevant_ids)

    # Round info
    round_num = len(sess["rounds"]) + 1
    expanded_terms = sorted(new_query_vec.items(), key=lambda x: -x[1])[:10]

    # What's new in expanded query
    orig_terms = set(sess["original_query_vec"].keys())
    new_terms = [t for t, _ in expanded_terms if t not in orig_terms]

    round_summary = {
        "round": round_num,
        "n_relevant": len(relevant_ids),
        "n_non_relevant": len(non_rel_ids),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "expanded_terms": [{"term": t, "weight": round(w, 4)} for t, w in expanded_terms],
        "new_terms": new_terms[:5],
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }

    sess["rounds"].append(round_summary)
    sess["query_vec"] = new_query_vec
    sess["results"] = new_results
    sess["metrics_history"].append({"before": metrics_before, "after": metrics_after})

    return jsonify({
        "session_id": sid,
        "round": round_num,
        "results": _format_results(new_results, sess["feedback"]),
        "round_summary": round_summary,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "new_terms": new_terms,
        "expanded_query": idx.vec_to_query_str(new_query_vec, top_n=8),
    })


@app.route("/api/pseudo_rf", methods=["POST"])
def api_pseudo_rf():
    """Pseudo Relevance Feedback (auto top-K)."""
    body = request.get_json(silent=True) or {}
    sid   = body.get("session_id", "")
    top_k = int(body.get("top_k", 5))
    beta  = float(body.get("beta", 0.5))

    if sid not in _sessions:
        return jsonify({"error": "Session not found"}), 404

    sess = _sessions[sid]
    idx  = engine.get_index(DATA_PATH)

    old_results = sess["results"]
    new_query_vec = engine.pseudo_rf(sess["query_vec"], old_results, idx, top_k=top_k, beta=beta)
    new_results = idx.search_by_vec(new_query_vec, top_k=TOP_K)

    # Use auto-marked top-k as "relevant" for metrics
    auto_rel_ids = {r["doc_id"] for r in old_results[:top_k]}
    metrics_before = engine.compute_metrics(old_results, auto_rel_ids)
    metrics_after  = engine.compute_metrics(new_results, auto_rel_ids)

    expanded_terms = sorted(new_query_vec.items(), key=lambda x: -x[1])[:10]
    orig_terms = set(sess["original_query_vec"].keys())
    new_terms = [t for t, _ in expanded_terms if t not in orig_terms]

    sess["query_vec"] = new_query_vec
    sess["results"] = new_results

    return jsonify({
        "session_id": sid,
        "results": _format_results(new_results, sess.get("feedback", {})),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "expanded_query": idx.vec_to_query_str(new_query_vec, top_n=8),
        "new_terms": new_terms[:6],
        "expanded_terms": [{"term": t, "weight": round(w, 4)} for t, w in expanded_terms],
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    body = request.get_json(silent=True) or {}
    sid = body.get("session_id", "")
    if sid in _sessions:
        del _sessions[sid]
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    idx = engine.get_index(DATA_PATH)
    domain_counts = {}
    for d in idx.docs:
        dom = d.get("domain", "Unknown")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    return jsonify({
        "total_docs": len(idx.docs),
        "vocab_size": len(idx.vocab),
        "avg_dl": round(idx.avg_dl, 1),
        "domain_counts": domain_counts,
        "active_sessions": len(_sessions),
    })


def _format_results(results: list[dict], feedback: dict) -> list[dict]:
    out = []
    for r in results:
        doc = r["doc"]
        did = str(r["doc_id"])
        out.append({
            "doc_id": r["doc_id"],
            "rank": r["rank"],
            "score": r["score"],
            "title": doc.get("title", ""),
            "domain": doc.get("domain", ""),
            "query": doc.get("query", ""),
            "snippet": doc.get("document", "")[:300],
            "url": doc.get("url", ""),
            "word_count": doc.get("word_count", 0),
            "auto_relevance": doc.get("relevance", 1),
            "user_feedback": feedback.get(did, ""),
        })
    return out


if __name__ == "__main__":
    logger.info(f"Loading index from {DATA_PATH}...")
    engine.get_index(DATA_PATH)
    logger.info("Index ready.")
    app.run(debug=False, port=5051, threaded=True)

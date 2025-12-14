"""
LLM Evaluation Pipeline
Hybrid evaluation system for:
1. Relevance & Completeness
2. Hallucination / Faithfulness
3. Latency & Cost

Supports real-world RAG outputs with:
- vector_data
- vectors_used
- final_response
"""

import json
import time
import os
from typing import List, Dict, Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Configuration
# =========================

JUDGE_MODEL = "gpt-4.1-mini"

COST_PER_1K_PROMPT_TOKENS = 0.0001
COST_PER_1K_RESPONSE_TOKENS = 0.0002

RELEVANCE_THRESHOLD = 0.75
HALLUCINATION_THRESHOLD = 0.65
HALLUCINATION_RATIO_LIMIT = 0.30

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

try:
    client = OpenAI()
except Exception:
    client = None


# =========================
# Helpers
# =========================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# =========================
# Preprocessing (FIXED)
# =========================

def preprocess_data(chat_path: str, context_path: str) -> Dict[str, Any]:
    """
    Robust preprocessing that supports:
    - conversation_turns
    - vector_data
    - vectors_used
    - final_response
    """
    chat = load_json(chat_path) if os.path.exists(chat_path) else {}
    context = load_json(context_path)

    # ---------- User Query ----------
    if "conversation_turns" in chat:
        turns = chat["conversation_turns"]
        query = turns[-2]["message"]
        history = "\n".join(
            f"{t['role']}: {t['message']}" for t in turns[:-2]
        )
    else:
        query = chat.get("query", "")
        history = ""

    # ---------- AI Response ----------
    if (
        "data" in context
        and "sources" in context["data"]
        and "final_response" in context["data"]["sources"]
    ):
        response = " ".join(context["data"]["sources"]["final_response"])
    else:
        response = chat.get("response", "")

    # ---------- Context Chunks (ONLY vectors_used) ----------
    vector_data = context["data"]["vector_data"]
    used_ids = set(context["data"]["sources"]["vectors_used"])

    context_chunks = [
        v["text"]
        for v in vector_data
        if v["id"] in used_ids
    ]

    return {
        "query": query,
        "response": response,
        "history": history,
        "context_chunks": context_chunks
    }


# =========================
# Fast Heuristics
# =========================

def fast_relevance_score(query: str, response: str) -> float:
    embeddings = EMBED_MODEL.encode(
        [query, response], normalize_embeddings=True
    )
    return float(
        cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    )


def fast_hallucination_ratio(
    response: str, context_chunks: List[str]
) -> float:
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if not sentences or not context_chunks:
        return 0.0

    sent_embs = EMBED_MODEL.encode(sentences, normalize_embeddings=True)
    ctx_embs = EMBED_MODEL.encode(context_chunks, normalize_embeddings=True)

    hallucinated = 0
    for emb in sent_embs:
        if cosine_similarity([emb], ctx_embs).max() < HALLUCINATION_THRESHOLD:
            hallucinated += 1

    return hallucinated / len(sentences)


# =========================
# LLM Judge (Fallback)
# =========================

def judge_with_llm(
    query: str,
    response: str,
    history: str,
    context_chunks: str,
    metric: str
) -> Dict[str, Any]:

    if not client:
        return {
            "score": 0,
            "rationale": "LLM judge skipped (no API key).",
            "judge_latency_ms": 0,
            "judge_cost": 0
        }

    if metric == "relevance":
        system_prompt = (
            "You are an expert evaluator. Score relevance and completeness "
            "on a scale of 0 to 5."
        )
        user_prompt = (
            f"Conversation History:\n{history}\n\n"
            f"User Query:\n{query}\n\n"
            f"AI Response:\n{response}\n\n"
            "Return JSON: {\"score\": int, \"rationale\": str}"
        )
    else:
        system_prompt = (
            "You are an expert evaluator focused on factual accuracy."
        )
        user_prompt = (
            f"Context:\n{context_chunks}\n\n"
            f"AI Response:\n{response}\n\n"
            "Return JSON: {\"score\": int, \"rationale\": str}"
        )

    start = time.time()
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    latency = (time.time() - start) * 1000

    result = json.loads(completion.choices[0].message.content)

    cost = (
        (completion.usage.prompt_tokens / 1000) * COST_PER_1K_PROMPT_TOKENS +
        (completion.usage.completion_tokens / 1000) * COST_PER_1K_RESPONSE_TOKENS
    )

    return {
        "score": result["score"],
        "rationale": result["rationale"],
        "judge_latency_ms": round(latency, 2),
        "judge_cost": round(cost, 6)
    }


# =========================
# Performance Metrics
# =========================

def estimate_target_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    latency_ms = 1200 + len(data["response"]) * 0.4

    prompt_chars = len(data["query"] + data["history"] + "".join(data["context_chunks"]))
    response_chars = len(data["response"])

    prompt_tokens = prompt_chars // 4
    response_tokens = response_chars // 4

    cost = (
        (prompt_tokens / 1000) * COST_PER_1K_PROMPT_TOKENS +
        (response_tokens / 1000) * COST_PER_1K_RESPONSE_TOKENS
    )

    return {
        "estimated_latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "estimated_cost": round(cost, 6)
    }


# =========================
# Main Pipeline
# =========================

def run_pipeline(chat_file: str, context_file: str):
    print("=== Running LLM Evaluation Pipeline ===")

    data = preprocess_data(chat_file, context_file)

    results = {
        "query": data["query"],
        "response": data["response"],
        "evaluation": {}
    }

    # ---- Relevance ----
    rel_score = fast_relevance_score(data["query"], data["response"])

    if rel_score < RELEVANCE_THRESHOLD:
        relevance = judge_with_llm(
            data["query"], data["response"], data["history"], "", "relevance"
        )
    else:
        relevance = {
            "score": round(rel_score * 5, 2),
            "rationale": "High semantic similarity; LLM judge skipped.",
            "judge_latency_ms": 0,
            "judge_cost": 0
        }

    # ---- Hallucination ----
    halluc_ratio = fast_hallucination_ratio(
        data["response"], data["context_chunks"]
    )

    if halluc_ratio > HALLUCINATION_RATIO_LIMIT:
        faithfulness = judge_with_llm(
            data["query"],
            data["response"],
            data["history"],
            "\n".join(data["context_chunks"]),
            "faithfulness"
        )
    else:
        faithfulness = {
            "score": round((1 - halluc_ratio) * 5, 2),
            "rationale": "Response grounded in retrieved context.",
            "judge_latency_ms": 0,
            "judge_cost": 0
        }

    performance = estimate_target_metrics(data)

    results["evaluation"]["relevance_completeness"] = relevance
    results["evaluation"]["faithfulness"] = faithfulness
    results["evaluation"]["performance"] = performance

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete → evaluation_results.json")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    CHAT_FILE = "clean_chat.json"   # can be empty or minimal
    CONTEXT_FILE = "clean_context.json"

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. LLM judge may be skipped.")

    run_pipeline(CHAT_FILE, CONTEXT_FILE)

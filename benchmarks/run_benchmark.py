import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from legal_rag.rag.rag_factory import get_rag_engine
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of examples.")
    return data


def run_rag(questions: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    """Run RAG system over questions and collect answers/contexts."""
    load_dotenv()
    engine = get_rag_engine()
    results: List[Dict[str, Any]] = []
    subset = questions if limit is None else questions[:limit]

    for item in subset:
        q = item["question"]
        rag_response = engine.query(q, use_hybrid_search=True, use_reranking=True)
        contexts = [c.get("text", "") for c in rag_response.get("search_results", [])]
        results.append(
            {
                "question_id": item.get("question_id"),
                "question": q,
                "answer": rag_response.get("answer", ""),
                "contexts": contexts,
                "sources": rag_response.get("sources", []),
            }
        )
    return results


def calculate_citation_accuracy(results: List[Dict[str, Any]], gold_citations: List[List[Dict[str, Any]]]) -> float:
    """Simple heuristic: count gold citations whose document + article appear in answer text."""
    total = 0
    correct = 0
    for res, gold in zip(results, gold_citations):
        answer_text = (res.get("answer") or "").lower()
        for cite in gold:
            doc = (cite.get("document") or "").lower()
            article = (cite.get("article") or "").lower()
            if not doc and not article:
                continue
            total += 1
            if doc in answer_text and (article in answer_text if article else True):
                correct += 1
    return correct / total if total else 0.0


def calculate_refusal_rate(results: List[Dict[str, Any]]) -> float:
    refusal_markers = [
        "не найдено",
        "недостаточно данных",
        "нет информации",
        "не могу ответить",
        "ақпарат жеткіліксіз",
    ]
    if not results:
        return 0.0
    refused = 0
    for res in results:
        answer = (res.get("answer") or "").lower()
        if any(marker in answer for marker in refusal_markers):
            refused += 1
    return refused / len(results)


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark with RAGAS metrics.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/benchmark_dataset.json"),
        help="Path to benchmark dataset JSON.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions for a quick run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to store benchmark outputs.",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    questions = [item["question"] for item in data]
    ground_truth = [item.get("ground_truth_answer", "") for item in data]
    ground_truth_citations = [item.get("ground_truth_citations", []) for item in data]

    results = run_rag(data, limit=args.limit)

    eval_dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": ground_truth if args.limit is None else ground_truth[: args.limit],
        }
    )

    scores = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )

    citation_acc = calculate_citation_accuracy(results, ground_truth_citations if args.limit is None else ground_truth_citations[: args.limit])
    refusal_rate = calculate_refusal_rate(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"benchmark_results_{timestamp}.json"

    payload = {
        "dataset": str(args.dataset),
        "limit": args.limit,
        "metrics": {
            "ragas": scores,
            "citation_accuracy": citation_acc,
            "refusal_rate": refusal_rate,
        },
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Benchmark finished. Results saved to {out_path}")
    print(f"RAGAS scores: {scores}")
    print(f"Citation accuracy: {citation_acc:.3f} | Refusal rate: {refusal_rate:.3f}")


if __name__ == "__main__":
    main()

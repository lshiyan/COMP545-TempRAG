import json
import argparse
import random
from time import time
from collections import defaultdict

from model.agent import QueryAgent
from model.reranker import Reranker
from model.search import IndexSearch
from model.tools import get_retrieval_tool, get_final_answer_tool
from prompts.query_cot import QUERY_COT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MultiTQ query agent evaluator")

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to the questions JSON file"
    )

    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to the FAISS index file"
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the FAISS metadata JSON file"
    )

    parser.add_argument(
        "--similarity_top_k",
        type=int,
        default=10,
        help="Top-K results to retrieve via FAISS similarity search"
    )

    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=50,
        help="Top-K results to keep after reranking"
    )

    parser.add_argument(
        "--sample_by",
        type=str,
        default="any",
        choices=["answer_type", "time_level", "qtype", "qlabel", "any"],
        help="Field to group questions by before sampling"
    )

    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Number of sampled questions per group. If not provided, use all."
    )

    return parser.parse_args()


def sample_questions(questions: list[dict], field: str, n: int) -> list[dict]:
    """
    Sample questions based on the given field.

    If field == "any":
        Randomly sample n questions from the full dataset.

    Else:
        Sample n questions for EACH distinct value in that field.
    """
    if n is None:
        return questions  # nothing to sample

    if field == "any":
        k = min(n, len(questions))
        return random.sample(questions, k)

    groups = defaultdict(list)
    for q in questions:
        groups[q[field]].append(q)

    sampled = []
    for key, group in groups.items():
        k = min(n, len(group))
        sampled.extend(random.sample(group, k))

    return sampled


def main():
    args = parse_args()

    print("Loading questions...", flush=True)
    with open(args.questions, "r") as q:
        questions = json.load(q)

    print(f"Total questions loaded: {len(questions)}", flush=True)

    if args.sample_n:
        questions = sample_questions(
            questions,
            field=args.sample_by,
            n=args.sample_n
        )

    print(f"Total questions after sampling: {len(questions)}", flush=True)

    # group statistics
    group_field = args.sample_by
    group_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    def get_group_key(q):
        return q[group_field] if group_field != "any" else "all"

    # Init reranker + search + tools
    rr = Reranker()
    search = IndexSearch(args.index, args.metadata)

    retrieval_tool = get_retrieval_tool(
        search=search,
        rr=rr,
        similarity_top_k=args.similarity_top_k,
        rerank_top_k=args.rerank_top_k
    )

    final_answer_tool = get_final_answer_tool()
    tools = [retrieval_tool, final_answer_tool]

    query_agent = QueryAgent(tools, QUERY_COT_SYSTEM_PROMPT)

    # Process questions
    for i, question in enumerate(questions):
        try: 
            start_time = time()
            print(f"Processing question {i + 1} / {len(questions)}", flush=True)

            query = question["question"]
            gold_answers = question["answers"]

            answer = query_agent.run_agent(query)
            answer_dict = query_agent.evaluate_agent_answer(query, answer, gold_answers)

            correct = answer_dict["correct"]
            group_key = get_group_key(question)

            group_stats[group_key]["total"] += 1
            if correct == "YES":
                group_stats[group_key]["correct"] += 1
                print("CORRECT ANSWER", flush=True)
            else:
                print("INCORRECT ANSWER", flush=True)

            print("Result:")
            print(json.dumps(answer_dict, indent=2), flush=True)
            print(f"Time taken: {time() - start_time:.2f} seconds", flush=True)
            print("-" * 50, flush=True)
        except Exception as e:
            print(f"Error processing question {i + 1}: {e}", flush=True)
            print("-" * 50, flush=True)
            continue
            
    # Print group performance summary
    print("\n================= PERFORMANCE BY GROUP =================\n")

    for group, stats in group_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0.0

        print(f"Group: {group}")
        print(f"  Total Questions : {total}")
        print(f"  Correct Answers : {correct}")
        print(f"  Accuracy        : {acc:.2%}\n")

    print("===================== END OF RUN =====================", flush=True)


if __name__ == "__main__":
    print("Starting MultiTQ experiment...", flush=True)
    main()

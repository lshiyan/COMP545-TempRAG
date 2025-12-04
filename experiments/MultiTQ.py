import json
import argparse
import random
from model.agent import QueryAgent
from model.reranker import Reranker
from model.search import IndexSearch
from model.tools import get_retrieval_tool, get_final_answer_tool
from prompts.query_cot import QUERY_COT_SYSTEM_PROMPT
from collections import defaultdict

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
        help="Number of sampled questions to run. If not provided, use all."
    )

    return parser.parse_args()

def sample_questions(questions: list[dict], field: str, n: int) ->  list[dict]:
    """
    Sample questions based on the given field.

    If field == "any":
        Randomly sample n questions from the full dataset.

    Else:
        Sample n questions for EACH distinct value in that field.

    Args:
        questions (list[dict]): List of question dicts.
        field (str): One of ["answer_type", "time_level", "qtype", "qlabel", "any"].
        n (int): Number of samples to draw.

    Returns:
        list[dict]: Sampled questions.
    """

    if field == "any":
        k = min(n, len(questions))
        return random.sample(questions, k)

    groups = defaultdict(list)
    for q in questions:
        key = q[field]
        groups[key].append(q)

    sampled = []
    for key, group in groups.items():
        k = min(n, len(group))
        sampled.extend(random.sample(group, k))

    return sampled

def main():
    args = parse_args()

    with open(args.questions, "r") as q:
        questions = json.load(q)

    if args.sample_by and args.sample_n:
        questions = sample_questions(
            questions,
            field=args.sample_by,
            n=args.sample_n
        )
    
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

    for _, question in enumerate(questions):
        query = question["question"]
        gold_answers = question["answers"]

        answer = query_agent.run_agent(query)
        answer_dict = query_agent.evaluate_agent_answer(answer, gold_answers)
        
        print(answer_dict)
        if answer_dict["correct"]:
            print("Correct!")

if __name__ == "__main__":
    main()

    
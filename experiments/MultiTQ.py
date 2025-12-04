import json
import argparse
from model.agent import QueryAgent
from model.reranker import Reranker
from model.search import IndexSearch
from model.tools import get_retrieval_tool, get_final_answer_tool
from prompts.query_cot import QUERY_COT_SYSTEM_PROMPT


def main(): 
    with open("data/MultiTQ/questions/full_questions.json", "r") as q:
        questions = json.load(q)

    rr = Reranker()
    search = IndexSearch("data/tkg/MultiTQ/full/full_index.faiss", "data/tkg/MultiTQ//full/full_metadata.json")
    
    retreval_tool = get_retrieval_tool(search, rr, 1, 50)
    final_answer_tool = get_final_answer_tool()
    
    tools = [retreval_tool, final_answer_tool]
    
    query_agent = QueryAgent(tools, QUERY_COT_SYSTEM_PROMPT)
    
    for i, question in enumerate(questions):
        query = question["question"]
        gold_answers = question["answers"]
        qtype = question["qtype"]
        
        answer = query_agent.run_agent(query)
        
        answer_dict = query_agent.evaluate_agent_answer(answer, gold_answers)
        
        print(answer_dict)
        if i == 3: break

if __name__ == "__main__":
    main()

    
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from model.search import IndexSearch
from dotenv import load_dotenv
from prompts.query_cot import QUERY_COT_FINAL_ANSWER_PROMPT
from typing import List
from model.reranker import Reranker
import os

load_dotenv()

#Has to be this way since IndexSearch is not serializable, tool arguments must be primitive.
def get_retrieval_tool(search: IndexSearch, rr: Reranker, similarity_top_k: int = 10, rerank_top_k: int = 50):
    
    @tool
    def retrieve_temporal_facts(query: str, constraints: dict, sorting: str = "") -> list:
        """
        Return a list of the top_k matches for the query, ordered by semantic similarity.
        
        Args:
            query (str): The query or subquery to answer. Must be a question.
            constraints (dict): A dictionary of constraints to filter the results by. Constraints can be 'before', 'after', or 'on'. If there are no constraints, input an empty dictionary.
            sorting (str): A string representing the type of sorting, if applicable. Sorting can be 'first', 'last'. If there are no sorts needed, input an empty string.
            similarity_top_k (int): The amount of documents to return after FAISS search.
            rerank_top_k (int): The amount of documents to keep after reranking.
        
        Returns:
            A list of the most relevant documents to the query. The format will be [match1, match2,]
        """
        results = search.search_index(query, constraints, sorting = sorting, rr = rr, similarity_top_k = similarity_top_k, rerank_top_k = rerank_top_k)

        return {'context': results}

    return retrieve_temporal_facts

def get_final_answer_tool():
    answer_llm = ChatOpenAI(model = os.getenv("PROCESSING_MODEL"))  # or any model you choose

    @tool
    def answer_from_context(query: str, context: List[str]) -> str:
        """
        Given the question and retrieved temporal context, output the final answer.
        
        Args:
            query (str): The query to answer.
            context (List[str]): A list of all existing context to answer the query.
        
        """
        context_str = "\n".join(context)

        prompt = QUERY_COT_FINAL_ANSWER_PROMPT.format(context_str = context_str, question = query)

        response = answer_llm.invoke(prompt)
        return response.content
    
    return answer_from_context
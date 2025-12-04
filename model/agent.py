from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from typing import List, Callable, Any
from const import get_openai_client
from prompts.answer_judge import JUDGE_ANSWER_PROMPT
import json
import os

load_dotenv()

FINAL_ANSWER = "Final Answer:"

class QueryAgentState(AgentState):
    context: List[str] = []
     
class QueryAgent():
    
    def __init__(self, tools: List[Callable], system_prompt: str):
        
        self.model_name = os.getenv("PROCESSING_MODEL")
        self.llm = ChatOpenAI(model = self.model_name)
        self.tools = tools
        self.system_prompt = system_prompt
        
        self.agent = self.build_agent()
    
    def build_agent(self) -> Any:
        """
        Creates a ReAct agent with the class variables.
        
        Returns:
            agent: The LangChain agent.
        """
        agent = create_agent(self.llm, self.tools, system_prompt = self.system_prompt, state_schema = QueryAgentState)
        return agent

    def run_agent(self, query: str) -> str:
        """
        Runs the agent with the given query.
        
        Args:
            query (str): The query to ask the agent to resolve.
        
        Returns:
            final_answer (str): A string representation of the final answer.
        """
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": []
        }

        final_answer = None
        for chunk in self.agent.stream(initial_state, stream_mode="updates"):

            """print("RAW CHUNK:", chunk)

            # 1. Check if context changed
            if "context" in chunk:
                print("=== CONTEXT UPDATED ===")
                print(chunk["context"])
                print("=======================\n")

            # 2. Standard model output
            if 'model' in chunk:
                msg =  chunk['model']['messages'][0].content
                print("MODEL:", chunk['model']['messages'][0].content)
                final_answer = msg

            # 3. Tool output
            if 'tools' in chunk:
                print("TOOL:", chunk['tools']['messages'][0].content)"""
            
        return final_answer.split(FINAL_ANSWER)[-1].strip()

    def evaluate_agent_answer(self, answer: str, gold_answers: list) -> dict:
        """
        Uses an LLM to judge whether the given answer is one of the gold answers.
        
        Args:
            answer (str): The answer the agent computes.
            gold_answer (list): A list of all the gold answers.
        
        Returns:
            response_dict (dict): A dict containing the answer, gold answers, and a correctness key which is <YES/NO>.
        """
        
        client = get_openai_client()
        
        response = client.responses.create(
            model = os.getenv("PROCESSING_MODEL"), 
            input = JUDGE_ANSWER_PROMPT.format(answer = answer, gold_answers= str(gold_answers))
        )
        
        responses_dict = response.output[1].content[0].text
        
        return responses_dict

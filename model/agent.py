from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from typing import List, Callable, Any
import json
import os

load_dotenv()

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

    def run_agent(self, query):
        """
        Runs the agent with the given query.
        
        Args:
            query (str): The query to ask the agent to resolve.
        """
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": []
        }

        for chunk in self.agent.stream(initial_state, stream_mode="updates"):

            print("RAW CHUNK:", chunk)

            # 1. Check if context changed
            if "context" in chunk:
                print("=== CONTEXT UPDATED ===")
                print(chunk["context"])
                print("=======================\n")

            # 2. Standard model output
            if 'model' in chunk:
                print("MODEL:", chunk['model']['messages'][0].content)

            # 3. Tool output
            if 'tools' in chunk:
                print("TOOL:", chunk['tools']['messages'][0].content)
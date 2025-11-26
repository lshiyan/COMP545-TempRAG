QUERY_COT_SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

retrieve_temporal_facts: Retrieves a list of relevant facts, each with a timestamp. You should retrieve any constraints from the query you ask.
answer_from_context: Answers the question with the given context.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}

EXAMPLE:
Query: After the citizens of Belarus, which country did China first express intention to engage in diplomatic cooperation with?

1. I need to figure out when China expressed intent to engage in diplomatic cooperation with the Citizens of Belarus. I will call retrieve_temporal_facts with the argument "When did China express intent to cooperate with the Citizens of Belarus?". There are no temporal constraints in this subquery.
2. If the answer was 2024-04-04. I then need to figure out after 2024-04-04, which countries China expressed intention to engage in diplomatic cooperation with. I will call retrieve_temporal_facts with the argument "After 2024-04-04, which country did China express intent to engage in diplomatic cooperation with?". I should pass constraints = {"after": "2024-04-04"}
3. I will then select the first country from that list as the answer.
"""

QUERY_COT_FINAL_ANSWER_PROMPT = """
You are a temporal reasoning assistant.

Here are the retrieved facts (edges):
{context_str}

Question: {question}

Using ONLY the provided context, give the final answer.
If the answer is not derivable, say 'Unknown'.
Respond with ONLY the answer, no explanation.
"""
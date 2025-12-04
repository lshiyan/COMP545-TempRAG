JUDGE_ANSWER_PROMPT = """
You are an evaluation model. Your task is to judge whether the model's final answer is correct.

You are given:

1. The model's final answer (a string).
2. A list of gold answers, where each item in the list is an acceptable correct answer string. 
   The gold answers may differ in formatting, capitalization, or textual paraphrasing, 
   but they all represent semantically correct ground truth answers.

RULES:

- If the model's final answer matches **any** of the gold answers after normalization, 
  the answer should be marked as YES.
- Normalization includes:
  * lowercasing,
  * trimming whitespace,
  * ignoring punctuation differences,
  * allowing date format variants that refer to the same point in time 
    (e.g., "2010-09-21" == "September 21, 2010").
- If the final answer does not appear to correspond to any item in the gold answer list, 
  it should be marked as NO.

OUTPUT FORMAT:

Return a JSON dictionary with the following fields:

{{
  "final_answer": "{answer}",
  "gold_answers": {gold_answers},
  "correct": "<YES or NO>"
}}

Be concise in your reasoning, and do not invent new information.

Answer: {answer}
Gold Answers: {gold_answers}
"""

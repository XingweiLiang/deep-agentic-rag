import os
from dotenv import load_dotenv
from rich.pretty import pprint as rprint

load_dotenv()
openai_api_key = os.getenv("OPEN_AI_KEY")

complex_query_adv = "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."


from graphs_node import planner_agent
print("Tool-Aware Planner Agent created successfully.")

# Test the planner agent
print("--- Testing Planner Agent ---")
test_plan = planner_agent.invoke({"question": complex_query_adv})
rprint(test_plan)


# Query Rewriting and Expansion: Using an LLM to transform naive sub-questions into high-quality search queries.

from graphs_node import query_rewriter_agent
print("Query Rewriter Agent created successfully.")

# Test the rewriter agent
print("--- Testing Query Rewriter Agent ---")
test_sub_q = test_plan.steps[2] # The synthesis step
test_past_context = "Step 1 Summary: NVIDIA's 10-K lists intense competition and rapid technological change as key risks. Step 2 Summary: AMD launched its MI300X AI accelerator in 2024 to directly compete with NVIDIA's H100."
rewritten_q = query_rewriter_agent.invoke({
    "sub_question": test_sub_q.sub_question,
    "keywords": test_sub_q.keywords,
    "past_context": test_past_context
})
print(f"Original sub-question: {test_sub_q.sub_question}")
print(f"Rewritten Search Query: {rewritten_q}")

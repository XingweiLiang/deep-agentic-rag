import os
from dotenv import load_dotenv
import json
from graphs_node import planner_agent
from graphs_node import policy_agent

load_dotenv()

# Define the complex query
complex_query_adv = "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."

# Create the test plan

test_plan = planner_agent.invoke({"question": complex_query_adv})

print("Policy Agent created.")

# Test the policy agent with different states
plan_str = json.dumps([s.dict() for s in test_plan.steps])
incomplete_history = "Step 1 Summary: NVIDIA's 10-K states that the semiconductor industry is intensely competitive and subject to rapid technological change."
decision1 = policy_agent.invoke({"question": complex_query_adv, "plan": plan_str, "history": incomplete_history})
print("--- Testing Policy Agent (Incomplete State) ---")
print(f"Decision: {decision1.next_action}, Justification: {decision1.justification}")

complete_history = incomplete_history + "\nStep 2 Summary: In 2024, AMD launched its MI300X accelerator to directly compete with NVIDIA in the AI chip market, gaining adoption from major cloud providers."
decision2 = policy_agent.invoke({"question": complex_query_adv, "plan": plan_str, "history": complete_history})
print("--- Testing Policy Agent (Complete State) ---")
print(f"Decision: {decision2.next_action}, Justification: {decision2.justification}")

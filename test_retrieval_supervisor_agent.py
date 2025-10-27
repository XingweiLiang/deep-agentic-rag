from dotenv import load_dotenv
load_dotenv()
import os

from graphs_node import retrieval_supervisor_agent
print("Retrieval Supervisor Agent created.")

# Test the supervisor
print("--- Testing Retrieval Supervisor Agent ---")
query1 = "revenue growth for the Compute & Networking segment in fiscal year 2023"
decision1 = retrieval_supervisor_agent.invoke({"sub_question": query1})
print(f"Query: '{query1}'")
print(f"Decision: {decision1.strategy}, Justification: {decision1.justification}")

query2 = "general sentiment about market competition and technological innovation"
decision2 = retrieval_supervisor_agent.invoke({"sub_question": query2})
print(f"Query: '{query2}'")
print(f"Decision: {decision2.strategy}, Justification: {decision2.justification}")

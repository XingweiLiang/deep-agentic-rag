from datasets import Dataset
import pandas as pd

def build_eval_dataset(baseline_retriever, baseline_result, final_state, complex_query_adv):
        
    print("Preparing evaluation dataset...")
    ground_truth_answer_adv = "NVIDIA's 2023 10-K lists intense competition and rapid technological change as key risks. This risk is exacerbated by AMD's 2024 strategy, specifically the launch of the MI300X AI accelerator, which directly competes with NVIDIA's H100 and has been adopted by major cloud providers, threatening NVIDIA's market share in the data center segment."

    # Retrieve context for the baseline model for the new query
    retrieved_docs_for_baseline_adv = baseline_retriever.invoke(complex_query_adv)
    baseline_contexts = [[doc.page_content for doc in retrieved_docs_for_baseline_adv]]

    # Consolidate all retrieved documents from all steps for the advanced agent
    advanced_contexts_flat = []
    for step in final_state['past_steps']:
        advanced_contexts_flat.extend([doc.page_content for doc in step['retrieved_docs']])
    advanced_contexts = [list(set(advanced_contexts_flat))] # Use set to remove duplicates for a cleaner eval

    eval_data = {
        'question': [complex_query_adv, complex_query_adv],
        'answer': [baseline_result, final_state['final_answer']],
        'contexts': baseline_contexts + advanced_contexts,
        'ground_truth': [ground_truth_answer_adv, ground_truth_answer_adv]
    }
    eval_dataset = Dataset.from_dict(eval_data)

    return eval_dataset
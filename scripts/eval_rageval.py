import os
import json
import logging
import subprocess
import sys
from typing import List, Dict
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag_service import rag_query
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def run_rag_generation(input_jsonl: str, output_rageval_jsonl: str):
    """
    Reads testset, generates predictions via RAG, and saves in RAGEval input format.
    RAGEval expected format (inferred from main.py):
    {
      "query": {"content": "..." },
      "prediction": {"content": "..." }, 
      "ground_truth": {"content": "...", "keypoints": [...] },
      "language": "vi",  # or "zh"/"en"
      ...
    }
    """
    logger.info(f"Generating predictions from {input_jsonl}...")
    
    if not os.path.exists(input_jsonl):
        logger.error(f"Input file not found: {input_jsonl}")
        return

    rag_inputs = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rag_inputs.append(json.loads(line))

    rageval_data = []
    for item in rag_inputs:
        q_content = item.get("query", {}).get("content")
        if not q_content:
            continue
            
        logger.info(f"Querying: {q_content}")
        try:
            # Call RAG service
            res = rag_query(question=q_content)
            answer_content = res.get("answer", "")
            
            # Construct RAGEval item
            # Preserve original fields, add prediction
            new_item = item.copy()
            new_item["prediction"] = {"content": answer_content}
            
            rageval_data.append(new_item)
            
        except Exception as e:
            logger.error(f"Error querying '{q_content}': {e}")

    # Save to intermediate file
    with open(output_rageval_jsonl, "w", encoding="utf-8") as f:
        for item in rageval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Saved {len(rageval_data)} items to {output_rageval_jsonl}")

def calculate_summary(output_file: str, summary_file: str):
    """
    Aggregates results similar to RAGEval's process_intermediate.py
    """
    logger.info("Calculating summary stats...")
    if not os.path.exists(output_file):
        logger.error(f"Output file not found: {output_file}")
        return

    data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Metrics to average
    metric_list = ['ROUGELScore', 'Precision', 'Recall', 'EIR', "completeness", "hallucination", "irrelevance"]
    sums = {m: 0.0 for m in metric_list}
    counts = {m: 0 for m in metric_list}

    for item in data:
        for m in metric_list:
            if m in item and item[m] is not None:
                sums[m] += float(item[m])
                counts[m] += 1
    
    avgs = {m: (sums[m] / counts[m] if counts[m] > 0 else 0.0) for m in metric_list if counts[m] > 0}
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(avgs, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Summary saved to {summary_file}")
    print("\n=== RAGEval Evaluation Summary ===")
    print(json.dumps(avgs, indent=4))
    print("==================================")

def run_rageval_metrics(input_file: str, output_file: str):
    """
    Calls RAGEval main.py via subprocess.
    """
    rageval_script = os.path.join(os.path.dirname(__file__), "..", "RAGEval", "rageval", "evaluation", "main.py")
    rageval_script = os.path.abspath(rageval_script)
    
    if not os.path.exists(rageval_script):
        logger.error(f"RAGEval script not found at {rageval_script}")
        return

    # Configuration for Groq as OpenAI
    groq_api_key = os.getenv("GROQ_API_KEY")
    os.environ["OPENAI_API_KEY"] = groq_api_key
    os.environ["BASE_URL"] = "https://api.groq.com/openai/v1"
    
    model_name = "llama-3.3-70b-versatile" 

    # Running ALL metrics: rouge-l, precision, recall, eir (local) AND keypoint_metrics (LLM)
    # Note: 'eir' (Entailment / Information Recall) sometimes requires model but RAGEval implementation might vary.
    # Looking at registry: rouge-l, precision, recall are standard str overlap. keypoint_metrics is LLM.
    metrics_to_run = "rouge-l precision recall keypoint_metrics"

    cmd = [
        sys.executable, rageval_script,
        "--input_file", input_file,
        "--output_file", output_file,
        "--metrics", "rouge-l", "precision", "recall", "keypoint_metrics",
        "--use_openai", 
        "--model", model_name,
        "--language", "en", # Using EN prompts as discussed
        "--version", "v1"
    ]
    
    logger.info(f"Running RAGEval command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        logger.info("RAGEval execution complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"RAGEval failed: {e}")

if __name__ == "__main__":
    # 1. Generate RAG Answer
    testset_path = r"d:\AI_Agent\data\testset.jsonl"
    rageval_input_path = r"d:\AI_Agent\data\rageval_input.jsonl"
    
    run_rag_generation(testset_path, rageval_input_path)
    
    # 2. Run Evaluation
    rageval_output_path = r"d:\AI_Agent\data\rageval_results.jsonl"
    run_rageval_metrics(rageval_input_path, rageval_output_path)

    # 3. Calculate Summary
    rageval_summary_path = r"d:\AI_Agent\data\rageval_summary.json"
    calculate_summary(rageval_output_path, rageval_summary_path)

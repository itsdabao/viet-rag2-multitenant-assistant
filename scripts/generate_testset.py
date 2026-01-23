import os
import json
import logging
import uuid
import time
from typing import List, Dict

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

def generate_testset():
    """
    Generates a RAGEval-compliant dataset (JSONL) from the knowledge base
    using Groq (Llama 3 / DeepSeek).
    """
    # 1. Configuration
    model_name = "llama-3.3-70b-versatile" # Robust for generation
    # Or use the user's tested one: "openai/gpt-oss-120b"
    
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in env.")
            return

        llm = Groq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0.2)
    except Exception as e:
        logger.error(f"Failed to init Groq: {e}")
        return

    input_file = r"d:\AI_Agent\data\knowledge_base\general_concepts.md"
    output_file = r"d:\AI_Agent\data\testset.jsonl"

    logger.info(f"Reading knowledge base from: {input_file}")
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Prompt for generation (RAGEval Schema)
    prompt = (
        f"You are an expert evaluator creating a test dataset for a RAG system.\n"
        f"Input Document:\n"
        f"--- START ---\n{content}\n--- END ---\n\n"
        f"Task: Generate 5 diverse and high-quality QA pairs based on the document above. For each question, extract key factual points (keypoints) from the answer.\n"
        f"Requirements:\n"
        f"1. Domain: 'Finance' (or 'Education' if more appropriate, but keeping schema consistent).\n"
        f"2. Language: 'vi' (Vietnamese) since the input document is in Vietnamese/English mix.\n"
        f"3. Format: Return a list of JSON objects. Do NOT use markdown code blocks.\n"
        f"4. Each object MUST match this schema exactly:\n"
        f"   {{\n"
        f"      'domain': 'Education',\n"
        f"      'language': 'vi',\n"
        f"      'query': {{ 'content': 'Question text here?', 'query_type': 'Factual' }},\n"
        f"      'ground_truth': {{ 'content': 'Detailed answer text.', 'keypoints': ['Point 1', 'Point 2'] }}\n"
        f"   }}\n"
        f"\n"
        f"Output pure JSON list."
    )

    # 3. Generate
    logger.info(f"Generating testset with {model_name}...")
    try:
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Cleanup markdown if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.startswith("```"):
             response_text = response_text.replace("```", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Parse list
        data_list = json.loads(response_text)
        
        logger.info(f"Generated {len(data_list)} items. Saving to JSONL...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data_list):
                # Enrich with IDs if missing
                if "query" in item:
                    if "query_id" not in item["query"]:
                        item["query"]["query_id"] = str(uuid.uuid4())
                
                # Write as single line JSON
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved RAGEval dataset to: {output_file}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON output: {e}\nResponse snippet: {response_text[:200]}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    generate_testset()

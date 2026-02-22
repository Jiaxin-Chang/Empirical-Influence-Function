import asyncio
import json
import os
import re
from openai import AsyncOpenAI

SYSTEM_PROMPT = """You are an expert Go programmer and data augmentation specialist.
Your task is to take a provided Go programming completion task (a user prompt and an assistant answer) and generate specialized training samples across 3 difficulty levels.

Level 1: Shallow Perturbation (浅层扰动)
- Positive: Keep the context and logic nearly identical, but slightly alter variable names, comments, or formatting.
- Negative: Keep the context identical, but introduce a subtle but critical logical bug (e.g., using the wrong specific variable, missing an error check).

Level 2: Deep Logical Rewrite (深层逻辑改写)
- Positive: Rewrite the user prompt to have a COMPLETELY DIFFERENT scenario/domain (e.g., from web server to file processing), but the core abstract programming pattern and logic must remain exactly the same. The assistant must provide the correct code for this new domain.
- Negative: Same as Level 2 Positive (different domain), but the assistant's logic contains a fundamental flaw in the core abstract pattern.

Level 3: Decoupled Knowledge / Sub-skill (解耦知识点)
- Positive: Do NOT solve the whole problem. Instead, create a task that teaches ONE decoupled, core sub-skill needed for the main problem (e.g., if the main problem freezes a config, teach just how to freeze an object in Go). 
- Negative: Create a task that teaches the wrong approach for that specific sub-skill.

Your output MUST be a valid JSON object matching this schema exactly:
{
    "variants": [
        {
            "perturbation_level": "Level_1",
            "variant_type": "positive",
            "generation_rationale": "Explanation of how this variant was generated...",
            "user_content": "Modified user prompt...",
            "assistant_content": "Modified correct assistant code..."
        },
        ... (repeat for Level_1 negative, Level_2 positive/negative, Level_3 positive/negative, making exactly 6 variants)
    ]
}
Do not include any Markdown formatting in your final output, just raw JSON. Give careful attention to escape characters in strings.
"""

USER_PROMPT_TEMPLATE = """Here is the original Go programming task.

--- ORIGINAL USER PROMPT ---
{original_user_content}

--- ORIGINAL ASSISTANT ANSWER ---
{original_assistant_content}

Generate the JSON with the 6 variants now."""


async def generate_variants(client: AsyncOpenAI, user_content: str, assistant_content: str, model: str = "gpt-5.2") -> dict:
    """Send original prompt and completion to GPT to get augmented positive/negative samples."""
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(
        original_user_content=user_content,
        original_assistant_content=assistant_content
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt},
        ],
        temperature=0.3,
    )
    
    content = response.choices[0].message.content or "{}"
    
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}\nRaw content:\n{content}")
        return {}

async def main():
    # 1. Provide API info (You can set ENV vars OPENAI_API_KEY and OPENAI_BASE_URL)
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.xi-ai.cn/v1"
    
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set. API calls might fail.")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # 2. Extract 1 sample from sft_test.jsonl
    test_file_path = "sft_test.jsonl"
    if not os.path.exists(test_file_path):
        print(f"File {test_file_path} not found.")
        return
        
    print(f"Loading first test sample from {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    sample = json.loads(first_line)
    
    # Parse out User and Assistant content
    original_user_content = ""
    original_assistant_content = ""
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            original_user_content = msg["content"]
        elif msg["role"] == "assistant":
            original_assistant_content = msg["content"]
            
    if not original_user_content or not original_assistant_content:
        print("Could not find user or assistant content in the first sample.")
        return
        
    print("--- Original Assistant Content ---")
    print(original_assistant_content)
    print("----------------------------------\n")
    
    # 3. Request Variants from LLM
    print("Sending construction request to GPT...")
    variants_response = await generate_variants(client, original_user_content, original_assistant_content)
    
    if "variants" in variants_response:
        variants = variants_response["variants"]
        out_file = "ground_truth_demo.jsonl"
        
        target_test_id = 0 # Using 0 since it's the first sample.
        
        with open(out_file, "w", encoding="utf-8") as f:
            for v in variants:
                level = v.get("perturbation_level", "Unknown")
                v_type = v.get("variant_type", "Unknown")
                rationale = v.get("generation_rationale", "")
                user_content = v.get("user_content", "")
                assistant_content = v.get("assistant_content", "")
                
                print(f"\n=== GENERATED {level.upper()} {v_type.upper()} ===")
                print(f"[Rationale]: {rationale}")
                print(f"[User Content Snippet]: {user_content[-200:]}")
                print("\n[Assistant Code]:")
                print(assistant_content)
                
                # Save to file
                sample_dict = {
                    "messages": [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ],
                    "format": "chatml",
                    "metadata": {
                        "target_test_id": target_test_id,
                        "perturbation_level": level,
                        "variant_type": v_type,
                        "generation_rationale": rationale
                    },
                    "label": f"{level}_{v_type}"
                }
                f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
                
        print(f"\nSaved {len(variants)} generated variants to {out_file}")
    else:
        print("Model did not return the expected 'variants' array.")

if __name__ == "__main__":
    asyncio.run(main())

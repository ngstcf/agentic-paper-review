import litellm
from litellm import completion, completion_cost
from typing import Dict, Any, Optional
import sys

# --- THIS WILL PRINT ONCE WHEN THE SCRIPT LOADS ---
print("\n" + "="*60)
print("--- LOADING core/llm_wrapper.py (V5 - EXPLICIT NONE FIX) ---")
print("="*60 + "\n")
sys.stdout.flush()
# --- END DEBUG BANNER ---

# Set LiteLLM to be verbose about errors
litellm.set_verbose = False

# Disable SSL verification globally for litellm
litellm.verify_ssl = False

# This is CRITICAL. It tells litellm to drop 'None' parameters.
litellm.drop_params = True 

def call_llm(
    prompt: str,
    system_prompt: str,
    provider: str,
    model: str, 
    temperature: float,
    max_retries: int
) -> Dict[str, Any]:
    """
    Unified LLM API call with retry logic, cost calculation,
    and explicit model-specific parameter handling.
    """
    model_string = f"{provider}/{model}"
    
    # --- EXPLICIT NONE FIX V5 ---
    
    print(f"\n[LLM_Wrapper_V5_DEBUG] Preparing call for model: '{model}'")
    
    # 1. Start with parameters common to ALL models
    params = {
        "num_retries": max_retries,
        "timeout": 120
    }

    # 2. Apply model-specific rules using an explicit if/else block
    if model.startswith("gpt-5"):
        # This is the block we EXPECT to run
        print("[LLM_Wrapper_V5_DEBUG] ---> MATCHED 'gpt-5'. Applying 'if' block.")
        
        # Rule 1: Temperature must be 1.0
        params["temperature"] = 1.0
        
        # Rule 2: Use *only* 'max_completion_tokens'
        params["max_completion_tokens"] = 4096 
        
        # --- THE CRITICAL FIX ---
        # Rule 3: Explicitly set max_tokens to None to prevent
        # litellm from adding its own default value.
        params["max_tokens"] = None
        # --- END CRITICAL FIX ---
        
    else:
        # This block is for all OTHER models (gpt-4o, claude, etc.)
        print(f"[LLM_Wrapper_V5_DEBUG] ---> DID NOT match 'gpt-5'. Applying 'else' block for '{model}'.")
        
        # Rule 1: Use the temperature from config
        params["temperature"] = temperature 
        
        # Rule 2: Use *only* 'max_tokens'
        params["max_tokens"] = 4096
    
    print(f"[LLM_Wrapper_V5_DEBUG] Final keys being sent to litellm: {list(params.keys())}")
    print(f"[LLM_Wrapper_V5_DEBUG] Value of max_tokens: {params.get('max_tokens')}")
    print(f"[LLM_Wrapper_V5_DEBUG] Value of max_completion_tokens: {params.get('max_completion_tokens')}")
    sys.stdout.flush() 
            
    # --- END: Explicit None Fix V5 ---

    try:
        response = completion(
            model=model_string,
            messages=[
                {"role": "system", "content": "system_prompt"},
                {"role": "user", "content": prompt}
            ],
            **params  # Unpack the dynamically built parameters
        )
        
        cost = completion_cost(completion_response=response)
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": response.usage.model_dump(),
            "cost": float(cost) if cost else 0.0
        }
    
    except Exception as e:
        print(f"[LLM_ERROR] Failed to call {model_string}: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": None,
            "cost": 0.0
        }
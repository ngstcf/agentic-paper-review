import litellm
from litellm import completion, completion_cost
from typing import Dict, Any, Optional
import sys
import os
import requests  # <-- NEW IMPORT
import json      # <-- NEW IMPORT

# --- THIS WILL PRINT ONCE WHEN THE SCRIPT LOADS ---
#print("\n" + "="*60)
#print("--- LOADING core/llm_wrapper.py (V8 - SURGICAL BYPASS FIX) ---")
#print("="*60 + "\n")
sys.stdout.flush()
# --- END DEBUG BANNER ---

# Set LiteLLM to be verbose about errors
litellm.set_verbose = False

# Disable SSL verification globally
litellm.verify_ssl = False
requests.packages.urllib3.disable_warnings() # Disable warnings for verify=False

# This is CRITICAL for all other calls
litellm.drop_params = True 

def _call_custom_ollama_bypass(
    system_prompt: str,
    prompt: str,
    model: str,
    response_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    This is a surgical bypass of litellm to call the custom_openai
    endpoint directly, using the parameters we know it requires.
    
    This function mimics the output format of litellm.
    """
    print(f"[LLM_Wrapper_V8_BYPASS] Executing surgical bypass for '{model}'")
    
    # 1. Get config from .env (which we set up in Step 1)
    OLLAMA_CHAT_URL = os.environ.get("CUSTOM_OPENAI_API_BASE")
    OLLAMA_API_KEY = os.environ.get("CUSTOM_OPENAI_API_KEY")

    if not OLLAMA_CHAT_URL or not OLLAMA_API_KEY:
        print("[LLM_Wrapper_V8_BYPASS] ERROR: CUSTOM_OPENAI_API_BASE or CUSTOM_OPENAI_API_KEY not set in .env")
        return {"success": False, "error": "Missing CUSTOM_OPENAI env variables", "content": None, "cost": 0.0}

    # 2. Set up headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }

    # 3. Define the *correct* payload with the *correct* parameters
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,               # Rule 1
        "max_completion_tokens": 16384,   # Rule 2 - Increased for longer reviews
        "stream": False
        # 'max_tokens' is (correctly) NOT included
    }

    # Add JSON mode if requested
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
        print(f"[LLM_Wrapper_V8_BYPASS] Enabling JSON mode")
    
    print(f"[LLM_Wrapper_V8_BYPASS] Payload keys: {list(payload.keys())}")

    # 4. Make the request
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=headers,
            json=payload,
            verify=False,  # Disable SSL verification
            timeout=300
        )
        
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # 5. Parse the response and format it like a litellm response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
        
        print("[LLM_Wrapper_V8_BYPASS] Success. Parsed response.")
        
        return {
            "success": True,
            "content": content.strip(),
            "usage": usage,
            "cost": 0.0 # We can't calculate cost for this bypass
        }
        
    except requests.exceptions.HTTPError as http_err:
        print(f"[LLM_Wrapper_V8_BYPASS] HTTP error: {http_err} - Response: {response.text}")
        return {"success": False, "error": f"HTTPError: {http_err} - {response.text}", "content": None, "cost": 0.0}
    except (Exception, KeyError, IndexError) as e:
        print(f"[LLM_Wrapper_V8_BYPASS] Error parsing response: {e} - Data: {data}")
        return {"success": False, "error": f"ParseError: {e}", "content": None, "cost": 0.0}


def call_llm(
    prompt: str,
    system_prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_retries: int,
    response_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified LLM API call with retry logic, cost calculation,
    and a surgical bypass for the 'custom_openai/gpt-5' combination.
    """
    model_string = f"{provider}/{model}"

    # --- NEW SURGICAL BYPASS V8 ---

   # print(f"\n[LLM_Wrapper_V8_DEBUG] Checking for bypass for provider: '{provider}', model: '{model}'")

    # Check for the *one* failing combination
    if provider.strip().lower() == "custom_openai" and model.strip().lower().startswith("gpt-5"):
        # ---
        # BLOCK 1: THE BYPASS
        # ---
        return _call_custom_ollama_bypass(
            system_prompt=system_prompt,
            prompt=prompt,
            model=model.strip(), # Pass the original model name
            response_format=response_format
        )
    
    # ---
    # BLOCK 2: ALL OTHER (WORKING) CALLS
    # ---
    #print(f"[LLM_Wrapper_V8_DEBUG] ---> No bypass. Proceeding with standard litellm call.")
    
    # 1. Start with standard params
    # Use higher max_tokens for newer models with larger context windows
    # deepseek-reasoner: up to 128k output tokens
    # gemini: up to 8k output tokens
    # gpt-4: up to 4k/16k output tokens
    provider_lower = provider.strip().lower()
    model_lower = model.strip().lower()

    # Set token limits based on model capabilities
    if "reasoner" in model_lower or "gpt-4" in model_lower or "gemini-2.5" in model_lower:
        max_tokens_limit = 32768  # Higher limit for reasoning models
    else:
        max_tokens_limit = 16384  # Standard limit for other models

    params = {
        "num_retries": max_retries,
        "timeout": 300,  # Increased timeout for longer responses (5 minutes)
        "temperature": temperature,
        "max_tokens": max_tokens_limit
    }

    # 2. Handle the *working* 'openai/gpt-5' variant
    if provider.strip().lower() == "openai" and model.strip().lower().startswith("gpt-5"):
        print("[LLM_Wrapper_V8_DEBUG] Applying 'openai/gpt-5' temperature override.")
        params["temperature"] = 1.0

    # 3. Add JSON mode if requested
    if response_format == "json":
        # Different providers have different JSON mode syntax
        provider_lower = provider.strip().lower()

        if provider_lower in ["openai", "custom_openai"]:
            # OpenAI and compatible APIs
            params["response_format"] = {"type": "json_object"}
            print(f"[LLM] Enabling JSON mode for {provider}/{model}")
        elif provider_lower == "anthropic":
            # Anthropic Claude
            params["response_format"] = {"type": "json_object"}
            print(f"[LLM] Enabling JSON mode for {provider}/{model}")
        elif provider_lower == "google":
            # Google Gemini - different format
            params["response_format"] = "json_object"
            print(f"[LLM] Enabling JSON mode for {provider}/{model}")
        # DeepSeek and others may not support JSON mode explicitly
        # They rely on prompt engineering instead
        elif provider_lower == "deepseek":
            print(f"[LLM] DeepSeek JSON mode via prompt engineering (no native JSON mode)")
        else:
            print(f"[LLM] JSON mode requested but {provider} support unknown, using prompt only")

    #print(f"[LLM_Wrapper_V8_DEBUG] Final litellm keys: {list(params.keys())}")

    try:
        response = completion(
            model=model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **params
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
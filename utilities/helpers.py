import logging
import re
import os
import hashlib
import json
from typing import Dict, Any
from core.config_loader import Config


def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def sanitize_model_name(model_string: str) -> str:
    """
    Sanitizes a model string (e.g., 'openai/gpt-4o') to be safe for filenames.
    Replaces slashes and other problematic characters with underscores.
    """
    # Replace slashes, colons, and backslashes with underscores
    sanitized = re.sub(r'[\/\\:]', '_', model_string)
    # Remove any other characters that are problematic in filenames
    sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
    return sanitized

def get_weights_table(criteria: list) -> str:
    """Generates a markdown table of criteria and their weights."""
    table = "| Criterion ID | Name | Weight |\n|--------------|------|--------|\n"
    for criterion in criteria:
        table += f"| {criterion['id']} | {criterion['name']} | {criterion.get('weight', 0)} |\n"
    return table

def get_scale_definition(scale: dict) -> str:
    """Generates a text description of the scoring scale."""
    if not scale or not scale.get('labels'):
        return "No scale defined."
    
    labels = scale['labels']
    description = "Score as follows:\n"
    for i, label in enumerate(labels):
        description += f"- {i}: {label}\n"
    return description

def get_config_hash(config: 'Config') -> str:
    """Generate a hash of the LLM configuration to detect parameter changes."""
    llm_config = config.get_llm_config()
    
    # Create a normalized string representation of the config
    config_str = json.dumps(llm_config, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(config_str.encode()).hexdigest()

def get_judge_config_hash() -> str:
    """Generate a hash of the Judge LLM configuration."""
    judge_config = {
        "provider": os.environ.get("JUDGE_PROVIDER", "openai"),
        "model": os.environ.get("JUDGE_MODEL", "gpt-4o"),
        "temperature": float(os.environ.get("JUDGE_TEMPERATURE", 0.1))
    }
    
    config_str = json.dumps(judge_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()
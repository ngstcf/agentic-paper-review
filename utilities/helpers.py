import logging
import re
import os
import hashlib
import json
import yaml
from typing import Dict, Any
from core.config_loader import Config, MODEL_DEFAULTS


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
        "provider": os.environ.get("JUDGE_PROVIDER", MODEL_DEFAULTS["judge_provider"]),
        "model": os.environ.get("JUDGE_MODEL", MODEL_DEFAULTS["judge_model"]),
        "temperature": float(os.environ.get("JUDGE_TEMPERATURE", 0.1))
    }

    config_str = json.dumps(judge_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Dictionary containing the parsed YAML content

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_novelty_adjusted_score(
    base_score: float,
    extractions: list,
    base_factor: float = 0.025,
    contradiction_penalty: float = 0.05,
    extension_bonus: float = 0.03
) -> float:
    """
    Adjust an overall score based on novelty rankings from extractions.

    Shared implementation used by both the Critic agent and the literature
    pipeline wrapper in run_review_with_dir_literature.py.

    Args:
        base_score: The base calculated score
        extractions: List of extraction objects with novelty_ranking,
                     contradicts_baseline, and extends_baseline attributes
        base_factor: Multiplier per novelty point above/below midpoint (3)
        contradiction_penalty: Subtracted from adjustment when contradictions found
        extension_bonus: Added to adjustment when extensions of baseline found

    Returns:
        Novelty-adjusted score clamped to [0, 100]
    """
    if not extractions:
        return base_score

    # Calculate average novelty ranking
    novelty_scores = [getattr(e, 'novelty_ranking', 3) for e in extractions]
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 3

    # Novelty adjustment factor (1-5 scale maps to -X% to +X% adjustment)
    adjustment_factor = (avg_novelty - 3) * base_factor

    # Check for contradictions (penalty)
    has_contradictions = any(getattr(e, 'contradicts_baseline', False) for e in extractions)
    if has_contradictions:
        adjustment_factor -= contradiction_penalty

    # Check for significant extensions (bonus)
    has_extensions = any(getattr(e, 'extends_baseline', False) for e in extractions)
    if has_extensions:
        adjustment_factor += extension_bonus

    # Apply adjustment
    adjusted_score = base_score * (1 + adjustment_factor)

    # Clamp to valid range
    return max(0, min(100, adjusted_score))
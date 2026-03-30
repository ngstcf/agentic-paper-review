# core/config_loader.py (updated)
import yaml
import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# --- ADD DEFAULT THRESHOLDS AS A FALLBACK ---
DEFAULT_THRESHOLDS = [
    {'threshold': 85, 'label': "Accept"},
    {'threshold': 70, 'label': "Accept with Revisions"},
    {'threshold': 50, 'label': "Revise and Resubmit"},
    {'threshold': 0, 'label': "Reject"}
]
# --- END OF ADDITION ---

class Config:
    def __init__(self, config_path="config"):
        self.config_path = config_path
        
        # Load global API keys from root .env first
        root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(root_env_path):
            load_dotenv(root_env_path)
            print(f"[Config] Loaded global API keys from {root_env_path}")
        
        # Load run-specific .env (for LLM parameters)
        run_env_path = os.path.join(config_path, ".env")
        if os.path.exists(run_env_path):
            print(f"[Config] Loading run-specific parameters from {run_env_path}")
            load_dotenv(run_env_path, override=True)  # Override global with run-specific
            print(f"[Config] Environment variables after loading .env:")
            # Debug: Print key environment variables
            key_vars = ["PROVIDER_EXTRACTION", "EXTRACTOR_MODEL", "PROVIDER_SYNTHESIS", "SYNTHESIZER_MODEL"]
            for key in key_vars:
                print(f"   {key}: {os.environ.get(key, 'not set')}")
        
        self.env = os.environ
        
        # Load criteria
        self.criteria_config = self._load_yaml(os.path.join(config_path, "criteria.yaml"))
        self.criteria = self.criteria_config.get('criteria', [])
        self.domain = self.criteria_config.get('domain', 'general academic')
        
        # --- ADD LOGIC TO LOAD, SORT, AND STORE THRESHOLDS ---
        self.recommendation_thresholds = self.criteria_config.get('recommendation_thresholds', DEFAULT_THRESHOLDS)
        # Sort by threshold, highest first. This is crucial for the logic.
        self.recommendation_thresholds.sort(key=lambda x: x.get('threshold', 0), reverse=True)
        # --- END OF ADDITION ---

        # Load prompts
        self.prompts = {
            "extractor_system": self._load_prompt("extractor_system.txt"),
            "extractor_user": self._load_prompt("extractor_user.txt"),
            "synthesizer_system": self._load_prompt("synthesizer_system.txt"),
            "synthesizer_user": self._load_prompt("synthesizer_user.txt"),
        }

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_prompt(self, file_name: str) -> str:
        with open(os.path.join(self.config_path, "prompts", file_name), 'r') as f:
            return f.read()

    def get_env(self, key: str, default: Any = None) -> Any:
        return self.env.get(key, default)

    def get_criteria(self) -> List[Dict[str, Any]]:
        return self.criteria
    
    def get_criterion_by_id(self, criterion_id: str) -> Dict[str, Any]:
        for c in self.criteria:
            if c['id'] == criterion_id:
                return c
        return {}

    # --- ADD A GETTER METHOD FOR THE THRESHOLDS ---
    def get_recommendation_thresholds(self) -> List[Dict[str, Any]]:
        return self.recommendation_thresholds
    # --- END OF ADDITION ---

    def get_prompt(self, name: str) -> str:
        return self.prompts.get(name, "")

    def get_llm_config(self) -> Dict[str, Any]:
        return {
            "extractor_provider": self.get_env("PROVIDER_EXTRACTION", "openai"),
            "extractor_model": self.get_env("EXTRACTOR_MODEL", "gpt-4o-mini"),
            "synthesizer_provider": self.get_env("PROVIDER_SYNTHESIS", "openai"),
            "synthesizer_model": self.get_env("SYNTHESIZER_MODEL", "gpt-4o"),
            "temperature": float(self.get_env("TEMPERATURE", 0.2)),
            "max_retries": int(self.get_env("MAX_RETRIES", 3)),
            "max_parallel": int(self.get_env("MAX_PARALLEL_EXTRACTIONS", 5)),
            "judge_provider": self.get_env("JUDGE_PROVIDER", "deepseek"),
            "judge_model": self.get_env("JUDGE_MODEL", "deepseek-reasoner"),
            "judge_temperature": self.get_env("JUDGE_TEMPERATURE",0.1)
        }

    def get_agent_config(self) -> Dict[str, Any]:
        """Return agent-specific temperatures and retry settings from env."""
        return {
            "librarian_temperature": float(self.get_env("LIBRARIAN_TEMPERATURE", 0.3)),
            "librarian_summary_temperature": float(self.get_env("LIBRARIAN_SUMMARY_TEMPERATURE", 0.5)),
            "fact_checker_temperature": float(self.get_env("FACT_CHECKER_TEMPERATURE", 0.3)),
            "critic_temperature": float(self.get_env("CRITIC_TEMPERATURE", 0.6)),
            "librarian_pre_search_delay": float(self.get_env("LIBRARIAN_PRE_SEARCH_DELAY", 1.0)),
            "librarian_retry_delay": float(self.get_env("LIBRARIAN_RETRY_DELAY", 2.0)),
            "critic_max_json_retries": int(self.get_env("CRITIC_MAX_JSON_RETRIES", 3)),
            "fact_checker_max_retries": int(self.get_env("FACT_CHECKER_MAX_RETRIES", 2)),
        }

    def get_token_limits(self) -> Dict[str, int]:
        """Return model-specific token limits from env."""
        return {
            "gpt4o": int(self.get_env("TOKEN_LIMIT_GPT4O", 16384)),
            "gpt4": int(self.get_env("TOKEN_LIMIT_GPT4", 4096)),
            "reasoner": int(self.get_env("TOKEN_LIMIT_REASONER", 32768)),
            "gemini": int(self.get_env("TOKEN_LIMIT_GEMINI", 32768)),
            "default": int(self.get_env("TOKEN_LIMIT_DEFAULT", 8192)),
            "custom_openai": int(self.get_env("TOKEN_LIMIT_CUSTOM_OPENAI", 16384)),
        }

    def get_timeout_config(self) -> Dict[str, Any]:
        """Return LLM/API timeout settings from env."""
        return {
            "llm_timeout": int(self.get_env("LLM_TIMEOUT", 300)),
            "api_timeout": int(self.get_env("API_TIMEOUT", 30)),
        }

    def get_novelty_config(self) -> Dict[str, Any]:
        """Return novelty adjustment factors from literature_sources.yaml."""
        lit_config = self._load_yaml(
            os.path.join(self.config_path, "..", "literature_sources.yaml")
        ) if os.path.exists(os.path.join(self.config_path, "..", "literature_sources.yaml")) else {}

        critic_config = lit_config.get("critic", {})
        novelty_adj = critic_config.get("novelty_adjustment", {})
        return {
            "base_factor": float(novelty_adj.get("base_factor", 0.025)),
            "contradiction_penalty": float(novelty_adj.get("contradiction_penalty", 0.05)),
            "extension_bonus": float(novelty_adj.get("extension_bonus", 0.03)),
        }

    def get_system_config(self) -> Dict[str, Any]:
        """Return system-wide limits and batch settings from env."""
        return {
            "max_content_tokens": int(self.get_env("MAX_CONTENT_TOKENS", 100000)),
            "batch_max_workers": int(self.get_env("BATCH_MAX_WORKERS", 4)),
            "cost_warning_per_paper": float(self.get_env("COST_WARNING_PER_PAPER", 1.0)),
            "arxiv_min_request_interval": float(self.get_env("ARXIV_MIN_REQUEST_INTERVAL", 3.0)),
        }
# setup_run.py (updated)
import os
import shutil
import argparse
from pathlib import Path
from core.config_loader import MODEL_DEFAULTS

def setup_run_directory(run_dir: str, config_dir: str = "config"):
    """Set up a new run directory with the required structure."""
    run_path = Path(run_dir)
    
    # Create directory structure
    papers_dir = run_path / "papers"
    input_dir = run_path / "input"
    outputs_dir = run_path / "outputs"
    reports_dir = outputs_dir / "reports"
    reviews_dir = outputs_dir / "reviews"
    
    for directory in [papers_dir, input_dir, outputs_dir, reports_dir, reviews_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Copy config files
    if os.path.exists(config_dir):
        print(f"Copying config files from {config_dir} to {input_dir}")
        shutil.copy(f"{config_dir}/criteria.yaml", input_dir / "criteria.yaml")
        
        # Copy prompts directory
        prompts_src = Path(config_dir) / "prompts"
        prompts_dst = input_dir / "prompts"
        if prompts_src.exists():
            if prompts_dst.exists():
                shutil.rmtree(prompts_dst)
            shutil.copytree(prompts_src, prompts_dst)
        
        # Create a .env file with LLM PARAMETERS ONLY (no API keys)
        env_file = input_dir / ".env"
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(f"""# LLM Configuration Parameters
# ======================================
# API keys are loaded from the global .env file at the project root
# ======================================

# Extraction Configuration
PROVIDER_EXTRACTION={MODEL_DEFAULTS["extractor_provider"]}
EXTRACTOR_MODEL={MODEL_DEFAULTS["extractor_model"]}

# Synthesis Configuration
PROVIDER_SYNTHESIS={MODEL_DEFAULTS["synthesizer_provider"]}
SYNTHESIZER_MODEL={MODEL_DEFAULTS["synthesizer_model"]}

# General Parameters
TEMPERATURE=0.2
MAX_RETRIES=3
MAX_PARALLEL_EXTRACTIONS=5

# Judge Configuration
JUDGE_PROVIDER={MODEL_DEFAULTS["judge_provider"]}
JUDGE_MODEL={MODEL_DEFAULTS["judge_model"]}
JUDGE_TEMPERATURE=0.1
""")
    else:
        print(f"Warning: Config directory {config_dir} not found. You'll need to create config files manually.")
    
    print(f"Run directory set up at: {run_path}")
    print(f"Please place your papers in: {papers_dir}")
    print(f"You can modify LLM parameters in: {input_dir / '.env'}")
    print(f"API keys are loaded from the global .env file at the project root")

def main():
    parser = argparse.ArgumentParser(description="Set up a new run directory")
    parser.add_argument("--run-dir", required=True, help="Directory for this run")
    parser.add_argument("--config-dir", default="config", help="Source config directory")
    args = parser.parse_args()
    
    setup_run_directory(args.run_dir, args.config_dir)

if __name__ == "__main__":
    main()
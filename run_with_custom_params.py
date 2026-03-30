# run_with_custom_params.py (fixed version)
import os
import argparse
from pathlib import Path
import subprocess
import time
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Run the review system with custom LLM parameters\n\n"
                    "DEFAULT: Standard review mode (NO literature grounding)\n"
                    "To enable literature features, use --literature-grounding flag.\n\n"
                    "Script selection:\n"
                    "  --literature-grounding present → run_review_with_dir_literature.py\n"
                    "  --literature-grounding absent  → run_review_with_dir.py (standard)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, help="Directory for this run")
    parser.add_argument("--provider-extraction", help="Extraction provider")
    parser.add_argument("--extractor-model", help="Extraction model")
    parser.add_argument("--provider-synthesis", help="Synthesis provider")
    parser.add_argument("--synthesizer-model", help="Synthesizer model")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--max-retries", type=int, help="Max retries")
    parser.add_argument("--max-parallel", type=int, help="Max parallel extractions")
    parser.add_argument("--judge-provider", help="Judge provider")
    parser.add_argument("--judge-model", help="Judge model")
    parser.add_argument("--judge-temperature", type=float, help="Judge temperature")
    parser.add_argument("--literature-grounding", action="store_true",
                        help="Enable literature grounding enhancement (default: DISABLED)")
    parser.add_argument("--web", action="store_true",
                        help="Enable web dashboard SSE backend")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔧 Configuring Custom LLM Parameters")
    print("=" * 80)
    
    # Update the .env file with the provided parameters
    env_file = Path(args.run_dir) / "input" / ".env"
    
    # Check if run directory exists
    if not env_file.parent.exists():
        print(f"❌ Error: Run directory '{args.run_dir}' does not exist!")
        print("   Please run 'python setup_run.py --run-dir <run_dir>' first.")
        return
    
    # Read the current .env file
    env_vars = {}
    if env_file.exists():
        print(f"📖 Reading existing configuration from {env_file}")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    else:
        print(f"⚠️  Warning: .env file not found at {env_file}, creating new one")
    
    # DEBUG: Print original values
    print(f"\n🔍 DEBUG: Original values before update:")
    print(f"   PROVIDER_EXTRACTION: {env_vars.get('PROVIDER_EXTRACTION', 'not set')}")
    print(f"   EXTRACTOR_MODEL: {env_vars.get('EXTRACTOR_MODEL', 'not set')}")
    print(f"   PROVIDER_SYNTHESIS: {env_vars.get('PROVIDER_SYNTHESIS', 'not set')}")
    print(f"   SYNTHESIZER_MODEL: {env_vars.get('SYNTHESIZER_MODEL', 'not set')}")
    
    # Track what parameters are being updated
    updated_params = []
    
    # Update with provided parameters (but NOT API keys)
    if args.provider_extraction:
        old_value = env_vars.get("PROVIDER_EXTRACTION", "not set")
        env_vars["PROVIDER_EXTRACTION"] = args.provider_extraction
        updated_params.append(f"PROVIDER_EXTRACTION: {old_value} → {args.provider_extraction}")
        print(f"🔧 DEBUG: Setting PROVIDER_EXTRACTION to {args.provider_extraction}")
        
    if args.extractor_model:
        old_value = env_vars.get("EXTRACTOR_MODEL", "not set")
        env_vars["EXTRACTOR_MODEL"] = args.extractor_model
        updated_params.append(f"EXTRACTOR_MODEL: {old_value} → {args.extractor_model}")
        print(f"🔧 DEBUG: Setting EXTRACTOR_MODEL to {args.extractor_model}")
        
    if args.provider_synthesis:
        old_value = env_vars.get("PROVIDER_SYNTHESIS", "not set")
        env_vars["PROVIDER_SYNTHESIS"] = args.provider_synthesis
        updated_params.append(f"PROVIDER_SYNTHESIS: {old_value} → {args.provider_synthesis}")
        print(f"🔧 DEBUG: Setting PROVIDER_SYNTHESIS to {args.provider_synthesis}")
        
    if args.synthesizer_model:
        old_value = env_vars.get("SYNTHESIZER_MODEL", "not set")
        env_vars["SYNTHESIZER_MODEL"] = args.synthesizer_model
        updated_params.append(f"SYNTHESIZER_MODEL: {old_value} → {args.synthesizer_model}")
        print(f"🔧 DEBUG: Setting SYNTHESIZER_MODEL to {args.synthesizer_model}")
        
    if args.temperature is not None:
        old_value = env_vars.get("TEMPERATURE", "not set")
        env_vars["TEMPERATURE"] = str(args.temperature)
        updated_params.append(f"TEMPERATURE: {old_value} → {args.temperature}")
        
    if args.max_retries is not None:
        old_value = env_vars.get("MAX_RETRIES", "not set")
        env_vars["MAX_RETRIES"] = str(args.max_retries)
        updated_params.append(f"MAX_RETRIES: {old_value} → {args.max_retries}")
        
    if args.max_parallel is not None:
        old_value = env_vars.get("MAX_PARALLEL_EXTRACTIONS", "not set")
        env_vars["MAX_PARALLEL_EXTRACTIONS"] = str(args.max_parallel)
        updated_params.append(f"MAX_PARALLEL_EXTRACTIONS: {old_value} → {args.max_parallel}")
        
    if args.judge_provider:
        old_value = env_vars.get("JUDGE_PROVIDER", "not set")
        env_vars["JUDGE_PROVIDER"] = args.judge_provider
        updated_params.append(f"JUDGE_PROVIDER: {old_value} → {args.judge_provider}")
        
    if args.judge_model:
        old_value = env_vars.get("JUDGE_MODEL", "not set")
        env_vars["JUDGE_MODEL"] = args.judge_model
        updated_params.append(f"JUDGE_MODEL: {old_value} → {args.judge_model}")
        
    if args.judge_temperature is not None:
        old_value = env_vars.get("JUDGE_TEMPERATURE", "not set")
        env_vars["JUDGE_TEMPERATURE"] = str(args.judge_temperature)
        updated_params.append(f"JUDGE_TEMPERATURE: {old_value} → {args.judge_temperature}")
    
    # DEBUG: Print updated values
    print(f"\n🔍 DEBUG: Values after update:")
    print(f"   PROVIDER_EXTRACTION: {env_vars.get('PROVIDER_EXTRACTION', 'not set')}")
    print(f"   EXTRACTOR_MODEL: {env_vars.get('EXTRACTOR_MODEL', 'not set')}")
    print(f"   PROVIDER_SYNTHESIS: {env_vars.get('PROVIDER_SYNTHESIS', 'not set')}")
    print(f"   SYNTHESIZER_MODEL: {env_vars.get('SYNTHESIZER_MODEL', 'not set')}")
    
    # Write the updated .env file (completely rewrite to avoid duplicates)
    print(f"💾 Saving updated configuration to {env_file}")
    with open(env_file, "w") as f:
        # Write header
        f.write(f"""# LLM Configuration Parameters for {args.run_dir}
# ======================================
# API keys are loaded from the global .env file at the project root
# ======================================
# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

""")
        
        # Write parameters in a specific order to avoid duplicates
        parameters_to_write = [
            ("PROVIDER_EXTRACTION", env_vars.get("PROVIDER_EXTRACTION", "openai")),
            ("EXTRACTOR_MODEL", env_vars.get("EXTRACTOR_MODEL", "gpt-4o-mini")),
            ("PROVIDER_SYNTHESIS", env_vars.get("PROVIDER_SYNTHESIS", "openai")),
            ("SYNTHESIZER_MODEL", env_vars.get("SYNTHESIZER_MODEL", "gpt-4o")),
            ("TEMPERATURE", env_vars.get("TEMPERATURE", "0.2")),
            ("MAX_RETRIES", env_vars.get("MAX_RETRIES", "3")),
            ("MAX_PARALLEL_EXTRACTIONS", env_vars.get("MAX_PARALLEL_EXTRACTIONS", "5")),
            ("JUDGE_PROVIDER", env_vars.get("JUDGE_PROVIDER", "openai")),
            ("JUDGE_MODEL", env_vars.get("JUDGE_MODEL", "gpt-4o")),
            ("JUDGE_TEMPERATURE", env_vars.get("JUDGE_TEMPERATURE", "0.1"))
        ]
        
        for key, value in parameters_to_write:
            if not key.endswith("_API_KEY"):  # Never write API keys to run-specific .env
                f.write(f"{key}={value}\n")
    
    # Verify the .env file was written correctly
    print(f"\n🔍 Verifying .env file content:")
    with open(env_file, "r") as f:
        content = f.read()
        for line in content.split('\n'):
            if line.strip() and not line.startswith("#"):
                print(f"   {line.strip()}")
    
    # Display configuration summary
    print("\n" + "=" * 80)
    print("📋 Configuration Summary")
    print("=" * 80)
    
    if updated_params:
        print("🔄 Updated parameters:")
        for param in updated_params:
            print(f"   • {param}")
    else:
        print("ℹ️  No parameters were updated")
    
    # Display current configuration
    print("\n🎯 Current LLM Configuration:")
    print(f"   Extraction: {env_vars.get('PROVIDER_EXTRACTION', 'openai')}/{env_vars.get('EXTRACTOR_MODEL', 'gpt-4o-mini')}")
    print(f"   Synthesis:  {env_vars.get('PROVIDER_SYNTHESIS', 'openai')}/{env_vars.get('SYNTHESIZER_MODEL', 'gpt-4o')}")
    print(f"   Temperature: {env_vars.get('TEMPERATURE', '0.2')}")
    print(f"   Max Retries: {env_vars.get('MAX_RETRIES', '3')}")
    print(f"   Max Parallel: {env_vars.get('MAX_PARALLEL_EXTRACTIONS', '5')}")
    
    if env_vars.get('JUDGE_PROVIDER') or env_vars.get('JUDGE_MODEL'):
        print(f"   Judge: {env_vars.get('JUDGE_PROVIDER', 'openai')}/{env_vars.get('JUDGE_MODEL', 'gpt-4o')}")
        print(f"   Judge Temperature: {env_vars.get('JUDGE_TEMPERATURE', '0.1')}")
    
    # Check if papers exist
    papers_dir = Path(args.run_dir) / "papers"
    paper_count = len(list(papers_dir.glob("*.pdf"))) + len(list(papers_dir.glob("*.md")))
    
    print(f"\n📁 Run Directory: {args.run_dir}")
    print(f"📄 Papers Found: {paper_count}")
    
    if paper_count == 0:
        print("\n⚠️  Warning: No papers found in the papers directory!")
        print("   Please add papers to the directory before running.")
        return
    
    # Check if there's existing progress
    progress_file = Path(args.run_dir) / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
            completed = len(progress_data.get("papers", {}))
            print(f"📊 Existing Progress: {completed} papers already completed")
        except:
            print("📊 Existing Progress: Unable to read progress file")
    else:
        print("📊 Existing Progress: No previous progress found")
    
    # Confirm before running
    print("\n" + "=" * 80)
    print("🚀 Starting Review Process")
    print("=" * 80)

    # Choose the appropriate script based on literature grounding flag
    if args.literature_grounding:
        script_name = "run_review_with_dir_literature.py"
        print("📚 Mode: Literature-Grounded Review Enhancement")
    else:
        script_name = "run_review_with_dir.py"
        print("📝 Mode: Standard Review")

    # Build command with appropriate flags
    cmd = ["python", script_name, "--run-dir", args.run_dir]
    if args.literature_grounding:
        cmd.append("--literature-grounding")
    if args.web:
        cmd.append("--web")

    # Set environment variables for this run
    env = os.environ.copy()
    env.update(env_vars)  # Use the updated env_vars

    print(f"🔧 Executing: {' '.join(cmd)}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Run the process and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env  # Use the modified environment
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        # Wait for the process to complete
        return_code = process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 80)
        if return_code == 0:
            print(f"✅ Process completed successfully!")
            print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        else:
            print(f"❌ Process failed with return code: {return_code}")
            print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user!")
        print("   Progress has been saved. You can resume by running the same command again.")
    except Exception as e:
        print(f"\n❌ Error running process: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
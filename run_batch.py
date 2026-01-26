#!/usr/bin/env python3
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import threading
import queue

import logging

# Configure logging to suppress LiteLLM messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

def stream_output(process, prefix=""):
    """Stream output from a subprocess in real-time."""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"{prefix}{line.rstrip()}")
    process.stdout.close()
    return_code = process.wait()
    return return_code

def run_single_directory(run_dir):
    """Run the review system for a single directory."""
    cmd = ["python", "run_review_with_dir.py", "--run-dir", run_dir]
    
    print(f"\n{'='*60}")
    print(f"🚀 Processing Directory: {run_dir}")
    print(f"{'='*60}")
    print(f"🔧 Executing: {' '.join(cmd)}")
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output
        return_code = stream_output(process, f"[{run_dir}] ")
        
        if return_code == 0:
            print(f"\n✅ Successfully completed {run_dir}")
            return True
        else:
            print(f"\n❌ Error running {run_dir} (return code: {return_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception processing {run_dir}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run batch review process")
    parser.add_argument("--parallel", action="store_true", help="Run directories in parallel")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()
    
    # List of run directories
    run_dirs = []
    i = 1
    while os.path.exists(f"run_dir{i}"):
        run_dirs.append(f"run_dir{i}")
        i += 1
    
    if not run_dirs:
        print("❌ No run directories found (run_dir1, run_dir2, etc.)")
        sys.exit(1)
    
    print("=" * 80)
    print(f"🚀 Starting Batch Review Process")
    print(f"📁 Found {len(run_dirs)} directories to process: {', '.join(run_dirs)}")
    if args.parallel:
        print(f"⚡ Running in parallel with {args.max_workers} workers")
    else:
        print(f"🔄 Running sequentially")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    
    if args.parallel:
        # Run in parallel
        print("\n⚡ Starting parallel processing...")
        
        def process_with_prefix(run_dir):
            return run_single_directory(run_dir), run_dir
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all jobs
            future_to_dir = {
                executor.submit(process_with_prefix, run_dir): run_dir 
                for run_dir in run_dirs
            }
            
            # Process as they complete
            completed = 0
            for future in as_completed(future_to_dir):
                run_dir = future_to_dir[future]
                completed += 1
                
                try:
                    success, _ = future.result()
                    if success:
                        successful_runs += 1
                    else:
                        failed_runs += 1
                except Exception as e:
                    print(f"\n❌ Exception processing {run_dir}: {e}")
                    failed_runs += 1
                
                print(f"\n📊 Batch Progress: {completed}/{len(run_dirs)} directories completed")
                
    else:
        # Run sequentially
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"\n📊 Batch Progress: {i}/{len(run_dirs)} directories")
            
            success = run_single_directory(run_dir)
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Batch Process Complete!")
    print("=" * 80)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📁 Total directories: {len(run_dirs)}")
    
    if failed_runs > 0:
        print(f"\n⚠️  {failed_runs} directories failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
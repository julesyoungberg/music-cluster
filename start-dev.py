#!/usr/bin/env python3
"""
Start Music Cluster development environment.
Starts both the FastAPI backend and Svelte frontend.
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def cleanup(processes):
    """Cleanup function to kill all child processes."""
    print(f"\n{BLUE}Shutting down...{NC}")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass

def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if virtual environment exists
    venv_python = script_dir / "venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"{RED}Error: Virtual environment not found.{NC}")
        print("Please create it first:")
        print("  python -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if node_modules exists
    ui_dir = script_dir / "ui"
    if not (ui_dir / "node_modules").exists():
        print(f"{BLUE}Installing Node.js dependencies...{NC}")
        subprocess.run(["npm", "install"], cwd=ui_dir, check=True)
    
    processes = []
    
    def signal_handler(sig, frame):
        cleanup(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start FastAPI server
        print(f"{GREEN}Starting FastAPI server on http://localhost:8000{NC}")
        api_proc = subprocess.Popen(
            [str(venv_python), "-m", "uvicorn", "music_cluster.api:app", "--reload", "--port", "8000"],
            cwd=script_dir
        )
        processes.append(api_proc)
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Start Svelte dev server
        print(f"{GREEN}Starting Svelte dev server on http://localhost:1420{NC}")
        ui_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ui_dir
        )
        processes.append(ui_proc)
        
        print(f"{GREEN}✓ Both servers are running!{NC}")
        print(f"{BLUE}API: http://localhost:8000{NC}")
        print(f"{BLUE}UI: http://localhost:1420{NC}")
        print(f"{BLUE}API Docs: http://localhost:8000/docs{NC}")
        print(f"\nPress Ctrl+C to stop both servers")
        
        # Wait for processes
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        cleanup(processes)
    except Exception as e:
        print(f"{RED}Error: {e}{NC}")
        cleanup(processes)
        sys.exit(1)

if __name__ == "__main__":
    main()

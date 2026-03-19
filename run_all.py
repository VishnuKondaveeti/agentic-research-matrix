import subprocess
import time
import sys
import os

def main():
    print("🚀 Starting Autonomous Multi-Agent Research System...")
    
    # Define commands
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    
    if not os.path.exists(python_exe):
        python_exe = "python" # Fallback if not using .venv
        
    api_cmd = [python_exe, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"]

    # Start FastAPI backend
    print("⏳ Starting FastAPI backend on port 8000...")
    api_process = subprocess.Popen(api_cmd)
    
    # Wait for API to initialize
    time.sleep(2)

    print("\n✅ System is running!")
    print("   - Web UI & API: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server.\n")

    try:
        # Keep the main thread alive
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        api_process.terminate()
        
        # Wait for it to close properly
        api_process.wait(timeout=5)
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()

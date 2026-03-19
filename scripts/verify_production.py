import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from rag.generator import Generator

def verify_setup():
    print("🔍 Verifying Production Readiness Modifications...")
    
    # 1. Check Configuration
    print(f"   - LLM Provider: {settings.llm_provider}")
    print(f"   - CORS Origins: {settings.cors_origins}")
    
    # 2. Check LLM Initialization (Dry Run)
    try:
        print(f"   - Attempting to initialize {settings.llm_provider} generator...")
        gen = Generator()
        print(f"     ✅ Generator initialized successfully with {settings.llm_provider}")
    except Exception as e:
        print(f"     ❌ Generator initialization failed: {e}")
        print("        (Note: This is expected if API keys or Ollama are not set up yet)")

    # 3. Check Directory Structure
    print("   - Checking runtime directories...")
    for d in [settings.papers_dir, settings.metadata_dir, settings.chroma_path, settings.logs_dir]:
        if d.exists():
            print(f"     ✅ {d.name} exists")
        else:
            print(f"     ⚠️  {d.name} missing (will be created on startup)")

    print("\n🚀 Verification complete!")

if __name__ == "__main__":
    verify_setup()

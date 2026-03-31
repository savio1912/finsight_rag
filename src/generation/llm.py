from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from langchain_ollama import ChatOllama
import httpx

def check_ollama_running():
    try:
        response = httpx.get("http://localhost:11434")
        if "Ollama" in response.text:
            print("Ollama is running.")
    except httpx.ConnectError:
        raise RuntimeError(
            "\nOllama is not running!\n"
            "Open a new terminal and run: ollama serve\n"
            "Then run this script again."
        )

def get_llm()->ChatOllama:
    """
    Load Mistral via Ollama.
    """
    print(f"Loading LLM '{LLM_MODEL}'...")
    check_ollama_running() 
    llm=ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_predict=LLM_MAX_TOKENS)
    print("LLM loaded.")
    return llm
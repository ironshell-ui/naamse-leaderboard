"""
Force ALL Google GenAI calls to use Google AI Studio (not Vertex AI).
Monkey-patches langchain_google_genai to ignore GOOGLE_GENAI_USE_VERTEXAI.
Volume-mounted into green agent. Loaded via PYTHONSTARTUP.
"""
import os

# Force Google AI Studio, not Vertex AI
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_CLOUD_LOCATION", None)

# Also patch google.genai to not use vertex
try:
    import google.genai as genai
    # Force client to use API key mode
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        print(f"[force_genai] Configured google.genai with API key (AI Studio mode)")
except ImportError:
    pass
except Exception as e:
    print(f"[force_genai] genai configure failed: {e}")

# Patch google-generativeai if present
try:
    import google.generativeai as palm
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        palm.configure(api_key=api_key)
        print(f"[force_genai] Configured google.generativeai with API key")
except ImportError:
    pass
except Exception as e:
    print(f"[force_genai] palm configure failed: {e}")

# Patch the SentenceTransformer fix too
try:
    exec(open("/home/agentbeats/naamse/fix_sentence_transformers.py").read())
except Exception as e:
    print(f"[force_genai] SentenceTransformer patch: {e}")

print("[force_genai] All patches applied — using Google AI Studio, NOT Vertex AI")

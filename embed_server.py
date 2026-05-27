from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
import sys
import json

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu');

while True:
    line = sys.stdin.readline()

    if not line:
        break

    try:
        data = json.loads(line)
        text = data.get("text", "")

        if not isinstance(text, str):
            raise ValueError(f"Expected string but got {type(text)}")

        text = text.strip();

        if not text:
            raise ValueError("Empty text")
        
        embedding = model.encode(text, show_progress_bar=False).tolist()

        print(json.dumps({
            "success": True,
            "embedding": embedding
        }), flush=True)
    
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }), flush=True)
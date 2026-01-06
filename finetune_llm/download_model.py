import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# Default model configuration
DEFAULT_MODEL_ID = "1TuanPham/T-VisStar-7B-v0.1"

model_id = os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
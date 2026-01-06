import os
from typing import List, Dict
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Check if running in remote mode (no local GPU needed)
LLM_API_URL = os.getenv("LLM_API_URL")

# Default model configuration
DEFAULT_MODEL_ID = "1TuanPham/T-VisStar-7B-v0.1"
DEFAULT_DEVICE = "cuda:0"


class RemoteLLM:
    """Remote LLM client that calls API endpoint (e.g., Google Colab)."""

    def __init__(self, api_url: str):
        import requests
        self._requests = requests
        self.api_url = api_url.rstrip('/')
        logger.info(f"Using remote LLM at {self.api_url}")

    def chat_complete(self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response by calling remote API."""
        try:
            response = self._requests.post(
                f"{self.api_url}/chat/complete",
                json={
                    "messages": messages,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except self._requests.exceptions.RequestException as e:
            logger.error(f"Remote LLM error: {e}")
            raise


class LocalLLM:
    """Local LLM that runs model on GPU."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = None, device: str = None):
        if self._initialized:
            return

        # Import heavy dependencies only when needed
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._torch = torch

        model_id = model_id or os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID)
        device = device or os.getenv("LLM_DEVICE", DEFAULT_DEVICE)

        self.model_id = model_id
        self.device = device

        logger.info(f"Loading model {model_id} on {device}...")

        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        logger.info(f"Model loaded successfully on {device}")
        self._initialized = True

    def chat_complete(self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate response from messages in OpenAI format.

        Args:
            messages: List of {"role": "system/user/assistant", "content": "..."}
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        # Format messages to chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


# Singleton instance
_llm_instance = None


def get_llm():
    """Get LLM instance - Remote if LLM_API_URL is set, otherwise Local."""
    global _llm_instance
    if _llm_instance is None:
        if LLM_API_URL:
            _llm_instance = RemoteLLM(LLM_API_URL)
        else:
            _llm_instance = LocalLLM()
    return _llm_instance


def get_local_llm() -> LocalLLM:
    """Backward compatible function - returns LLM instance (local or remote)."""
    return get_llm()


def local_chat_complete(messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Convenience function to generate chat completion - works with both local and remote LLM."""
    llm = get_llm()
    return llm.chat_complete(messages, max_new_tokens, temperature)

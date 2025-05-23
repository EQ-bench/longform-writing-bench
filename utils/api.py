# utils/api.py
import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List # Added List
from dotenv import load_dotenv
import re

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI, Anthropic, OpenRouter, etc.).
    Handles basic retry logic.
    """

    def __init__(self, model_type=None, request_timeout=None, max_retries=None, retry_delay=None):
        self.model_type = model_type or "default" # 'test' or 'judge'

        # Load config with fallback to env vars defined in this file
        default_timeout = 300
        default_retries = 3
        default_delay = 5

        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else: # Default/fallback
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(request_timeout if request_timeout is not None else os.getenv("REQUEST_TIMEOUT", default_timeout))
        self.max_retries = int(max_retries if max_retries is not None else os.getenv("MAX_RETRIES", default_retries))
        self.retry_delay = int(retry_delay if retry_delay is not None else os.getenv("RETRY_DELAY", default_delay))

        # Determine API provider for header/payload structure
        self.provider = "openai" # Default assumption
        if "anthropic.com" in self.base_url:
            self.provider = "anthropic"
        #elif "google" in self.base_url:
        #     self.provider = "google" # Gemini
        # Add more providers (Mistral, etc.) if their structure differs significantly

        self.headers = self._get_headers()

        logging.debug(f"Initialized {self.model_type} API client. Provider: {self.provider}, URL: {self.base_url}, Timeout: {self.request_timeout}")
        if not self.api_key:
            logging.warning(f"API Key for {self.model_type} is not set!")

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            # headers["anthropic-beta"] = "max-tokens-3-5-sonnet-2024-07-15" # Example for specific features if needed
        elif self.provider == "google":
             # Google via specific proxy might use OpenAI format Authorization
             headers["Authorization"] = f"Bearer {self.api_key}"
        else: # OpenAI, OpenRouter, Mistral (usually Bearer token)
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_payload(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, min_p: Optional[float] = None) -> Dict[str, Any]:
        if self.provider == "anthropic":
            # Anthropic expects messages differently, system prompt separate
            system_prompt = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    # Anthropic alternates user/assistant roles strictly
                    if not user_messages or user_messages[-1]["role"] != msg["role"]:
                         user_messages.append(msg)
                    else: # Merge consecutive messages of same role (e.g. multiple user messages)
                        user_messages[-1]["content"] += "\n\n" + msg["content"]

            payload = {
                "model": model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                payload["system"] = system_prompt
            # Anthropic doesn't use min_p directly, might have top_p, top_k
            # if min_p is not None: payload["top_p"] = 1.0 - min_p # Approximation if needed
        else: # OpenAI format
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "min_p": min_p
            }

            if self.base_url == 'https://api.openai.com/v1/chat/completions':
                del payload['min_p']
            if model == 'o3':
                del payload['max_tokens']
                payload['max_completion_tokens'] = max_tokens
                payload['temperature'] = 1

        return payload

    def _extract_content(self, data: Dict[str, Any]) -> str:
        try:
            if self.provider == "anthropic":
                if data.get("type") == "error":
                    logging.error(f"Anthropic API Error: {data.get('error', {}).get('type')} - {data.get('error', {}).get('message')}")
                    raise RuntimeError(f"Anthropic API Error: {data.get('error', {}).get('message')}")
                # Check if content is a list (newer API versions)
                if isinstance(data.get("content"), list):
                    # Find the first text block
                    text_block = next((block["text"] for block in data["content"] if block.get("type") == "text"), None)
                    if text_block:
                        return text_block
                    else:
                         logging.warning("Anthropic response content list did not contain a text block.")
                         return ""
                else:
                    # Fallback for older structure (if any)
                    return data.get("completion", "") # Or adjust based on actual response
            else: # OpenAI format
                content = data["choices"][0]["message"]["content"]
                # Strip common self-correction/thinking blocks
                content = self._strip_thinking_blocks(content)
                return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Error parsing API response content: {e}. Response data: {data}")
            raise RuntimeError(f"Could not extract content from API response: {e}") from e

    def _strip_thinking_blocks(self, content: str) -> str:
        """Removes common <thinking> or <reasoning> blocks."""
        patterns = [
            r"<thinking>.*?</thinking>\s*",
            r"<reasoning>.*?</reasoning>\s*",
            r"\[thinking\].*?\[/thinking\]\s*",
            r"\{thinking\}.*?\{/thinking\}\s*",
             # Add more patterns if needed
        ]
        cleaned_content = content
        for pattern in patterns:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL | re.IGNORECASE)
        return cleaned_content


    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = None, use_random_seed: bool = False) -> str:
        """
        Generates text using the configured API endpoint.
        'messages' should be in OpenAI format [{'role': 'user', 'content': '...'}].
        The method adapts the payload for different providers.
        """
        payload = self._prepare_payload(model, messages, temperature, max_tokens, min_p)


        # set a specific provider on openrouter if required
        if False and model in ['meta-llama/llama-4-scout', 'meta-llama/llama-4-maverick']:
            print('groqqing')
            payload['provider'] =  {
                "order": [
                    "DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                    #"Parasail",
                    #"Fireworks",
                    #"Mistral" # mistral-small-3
                    #"Lambda", # llama-3.1-8b
                    #"NovitaAI",  # qwen-72b, llama-3.1-8b
                    #"Nebius AI Studio", # qwen-72b
                    #"Hyperbolic", # qwen-72b
                    #"inference.net", # llama-3.1-8b
                    #"Groq", # llama 3.1 8b
                    #"inference.net",
                ],
                "allow_fallbacks": False
            }

        if use_random_seed:
            print('using random seed')
            seed_lines = [
                ''.join(random.choices(string.ascii_letters + string.digits, k=80)) for _ in range(5)
            ]
            random_seed_block = (
                "<RANDOM SEED PLEASE IGNORE>\n" +
                "\n".join(seed_lines) +
                "\n</RANDOM SEED>"
            )
            messages = [{"role": "system", "content": random_seed_block}] + messages

        if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
            if 'qwen3' in model.lower():
                # optionally disable thinking for qwen3 models
                print('/no_think')
                system_msg = [{"role": "system", "content": "/no_think"}]
                messages = system_msg + messages

        for attempt in range(self.max_retries):
            logging.debug(f"API Call Attempt {attempt+1}/{self.max_retries} to {self.model_type} model {model} via {self.base_url}")

            response = None # Initialize response variable
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                # logging.debug(f"Raw API Response Data: {json.dumps(data, indent=2)}") # Uncomment for deep debugging
                content = self._extract_content(data)
                if not content and self.provider == "anthropic" and data.get("stop_reason") == "max_tokens":
                     logging.warning(f"Anthropic response stopped due to max_tokens ({max_tokens}). Content might be incomplete.")

                logging.debug(f"API Call Successful. Received ~{len(content)} chars.")
                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out ({self.request_timeout}s) on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.RequestException as e: # Catches connection errors, HTTP errors, etc.
                status_code = e.response.status_code if e.response is not None else 'N/A'
                logging.error(f"API Request failed (Status: {status_code}) on attempt {attempt+1}/{self.max_retries}: {e}")
                if e.response is not None:
                    logging.error(f"Response Body: {e.response.text}")
                    # Handle rate limits specifically
                    if e.response.status_code == 429:
                        logging.warning("Rate limit exceeded. Backing off...")
                        # Implement exponential backoff or use Retry-After header if available
                        wait_time = self.retry_delay * (2 ** attempt) # Exponential backoff
                        logging.info(f"Waiting for {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue # Skip the standard delay and retry immediately after backoff
                    # Handle other specific errors if needed (e.g., 401 Unauthorized, 400 Bad Request)
                    elif e.response.status_code == 401:
                         logging.critical(f"Authentication failed for {self.model_type} API. Check API Key.")
                         raise RuntimeError(f"Authentication failed for {self.model_type} API.") from e
                    elif e.response.status_code == 400:
                         logging.error(f"Bad Request (400). Check payload structure/parameters for {self.provider} API.")
                         # Don't retry on bad request, it's likely a code issue
                         raise RuntimeError(f"Bad Request (400) for {self.model_type} API.") from e

            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries}. Response text: {response.text if response else 'No response object'}")
            except RuntimeError as e: # Catch errors raised by _extract_content or specific handlers
                 logging.error(f"Runtime error during API call processing on attempt {attempt+1}/{self.max_retries}: {e}")
                 # Decide if this specific runtime error is retryable
                 if "Authentication failed" in str(e) or "Bad Request" in str(e):
                     raise e # Don't retry these
            except Exception as e:
                logging.exception(f"An unexpected error occurred during API call attempt {attempt+1}/{self.max_retries}: {e}") # Use .exception to log stack trace

            # Wait before retrying (unless it was a rate limit backoff)
            if attempt < self.max_retries - 1:
                 logging.info(f"Waiting {self.retry_delay} seconds before next attempt...")
                 time.sleep(self.retry_delay)

        # If loop completes without returning, all retries failed
        raise RuntimeError(f"Failed to generate text with {self.model_type} model {model} after {self.max_retries} attempts.")
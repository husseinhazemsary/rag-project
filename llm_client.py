import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_SERVER = os.getenv("MODEL_SERVER", "NGU")

API_CONFIG = {
    "NGU": {
        "api_key": os.getenv("NGU_API_KEY"),
        "base_url": os.getenv("NGU_BASE_URL"),
        "model": os.getenv("NGU_MODEL")
    },
    "GROQ": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": os.getenv("GROQ_BASE_URL"),
        "model": os.getenv("GROQ_MODEL")
    }
}

config = API_CONFIG.get(MODEL_SERVER)

if not config or not config["api_key"]:
    raise ValueError(f"API Key for {MODEL_SERVER} is missing in .env file!")

def generate_response(prompt, max_tokens=300, temperature=0.3):
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }

    data = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(
        url=f"{config['base_url']}/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM API Error: {response.status_code} - {response.text}")

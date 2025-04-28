import requests

# Set your NVIDIA API key here
api_key = "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"

# Set to False since we're not using streaming
stream = False

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json"
}

# Customize your prompt here
user_prompt = "Write a short poem about the moon."

payload = {
    "model": "meta/llama-4-maverick-17b-128e-instruct",
    "messages": [
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    "max_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.95,
    "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    # Pretty print the response
    result = response.json()
    print(result['choices'][0]['message']['content'])

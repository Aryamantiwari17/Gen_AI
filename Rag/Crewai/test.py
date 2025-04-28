from crewai import LLM
from pydantic import BaseModel
import json
import re

class Dog(BaseModel):
    name: str
    age: int
    breed: str

llm = LLM(
    model="groq/mistral-saba-24b",
    temperature=0.7,
)

prompt = (
    "Extract structured info as JSON with keys 'name', 'age', and 'breed'. "
    "Respond *only* with a JSON block like:\n"
    '{ "name": "Kona", "age": 3, "breed": "black german shepherd" }'
    "\n\nMessage: Meet Kona! She is 3 years old and is a black german shepherd."
)

response = llm.call(prompt)

# 🧹 Clean up any markdown formatting or non-JSON parts
cleaned = re.sub(r"```json|```", "", response).strip()

# ✅ Parse into the Pydantic model
dog_data = Dog.parse_raw(cleaned)
print(dog_data)

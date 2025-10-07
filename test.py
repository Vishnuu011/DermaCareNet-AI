import os

# os.system(
#     'cd yolov5 && python detect.py --weights "C:\\Users\\VISHNU\\Desktop\\DermaCareNet\\DermaCareNet-AI\\models\\best.pt" --img 640 --conf 0.5 --source C:\\Users\\VISHNU\\Desktop\\DermaCareNet\\DermaCareNet-AI\\lnajw8usf40kl8x2f3gi.jpeg --save-txt'
# )

import base64
import mimetypes
from pathlib import Path

def local_image_to_data_url(image_path):
    # Ensure the file exists
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Guess MIME type (e.g., 'image/jpeg', 'image/png')
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback

    # Read and encode the file
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Return as data URL
    return f"data:{mime_type};base64,{encoded}"

# Usage
IMAGE_DATA_URL = local_image_to_data_url("download (2).png")
#print(IMAGE_DATA_URL)

from groq import Groq

from groq import Groq
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import requests
import mimetypes
import base64
from pathlib import Path
from pydantic import PrivateAttr


# ------------------------------
# 1. Groq Vision Wrapper (text + image)
# ------------------------------
class GroqVisionLLM(LLM):
    _client: Groq = PrivateAttr()
    model: str

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = Groq(api_key=api_key)

    @property
    def _llm_type(self):
        return "groq-vision-llm"

    def _call(self, prompt: str, image_url: str = None, stop=None):
        user_content = [{"type": "text", "text": prompt}]
        if image_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.7,
            max_completion_tokens=512
        )
        return completion.choices[0].message.content


# ------------------------------
# 2. Serper Tool (Google Search)
# ------------------------------
def serper_search(query: str):
    headers = {
        "X-API-KEY": "",
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 5}
    response = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    results = response.json()
    if "organic" in results:
        return "\n".join(
            [f"{item['title']} - {item['link']}" for item in results["organic"]]
        )
    return "No results found."

serper_tool = Tool(
    name="Google Search",
    func=serper_search,
    description="Useful for finding treatments, causes, or resources for skin conditions."
)


# ------------------------------
# 3. Initialize Groq + Agent
# ------------------------------
groq_vision = GroqVisionLLM(
    api_key="",
    model="meta-llama/llama-4-maverick-17b-128e-instruct"
)

tools = [serper_tool]

agent = initialize_agent(
    tools=tools,
    llm=groq_vision,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# ------------------------------
# 4. Run a Query
# ------------------------------
# query = f"""
# Analyze this facial skin condition image: {IMAGE_DATA_URL}.
# Return:
# 1. Possible skin condition(s) (Acne, Blackheads, Dark-Spots, Dry-Skin, Englarged-Pores, Eyebags, Oily-Skin, Skin-Redness, Whiteheads, Wrinkles etc.)
# 2. Common causes and triggers
# 3. Hygiene and lifestyle factors
# 4. Recommended foods to eat and foods to avoid
# 5. General over-the-counter or clinical treatment options
# 6. Complications if untreated
# 7. When to consult a dermatologist

# Then, search Google for the best treatment and diet resources.
# """
query = f"""
Analyze this skin image {IMAGE_DATA_URL}.
Return:
- Condition
- Causes
- Foods (eat/avoid)
- Treatments
- Doctor advice
"""

result = agent.run(query)
print("\n=== FINAL RESULT ===")
print(result)
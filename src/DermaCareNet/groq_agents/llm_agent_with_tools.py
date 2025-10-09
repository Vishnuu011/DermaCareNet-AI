import os, sys 
from typing import List, Tuple, Optional

from src.DermaCareNet.exception import ComputerVisionYolov11Exception
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import json
import requests
from dotenv import load_dotenv
load_dotenv()


@tool
def google_serpapi_search(query: str) -> str:
    """Searches the web for information using SerpApi."""
    url = "https://google.serper.dev/search"

    payload = json.dumps({
      "q": query
    })
    headers = {
      'X-API-KEY': os.environ["SERPAPI_API_KEY"],
      'Content-Type': 'application/json'
    }    

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"


def skin_disease_agent(model: str) -> Optional[AgentExecutor]:

    """
    Creates and returns a dermatology-focused AI agent that analyzes detected 
    skin conditions using Groq LLM and search tools.

    This agent is designed as part of DermaCareNet-AI and uses a Groq-hosted 
    language model to provide medical insights about skin diseases. It leverages 
    a search tool (e.g., Google SerpAPI) to gather information from trusted 
    dermatology sources before generating structured summaries.

    The generated summary includes:
        1. Condition overview
        2. Common causes
        3. Hygiene and prevention tips
        4. Diet recommendations
        5. Over-the-counter or natural treatments
        6. Possible complications if untreated
        7. When to consult a dermatologist

    Args:
        model (str): The name of the Groq model to be used (e.g., "mixtral-8x7b" or "llama2-70b-chat").

    Returns:
        Optional[AgentExecutor]: Configured LangChain AgentExecutor capable of 
        using search tools and generating dermatological reports. Returns None 
        if initialization fails.

    Raises:
        ComputerVisionYolov11Exception: If any initialization or runtime error 
        occurs during the setup of the Groq LLM or AgentExecutor.
    """

    try:
        llm = ChatGroq(
        temperature=0,
        model_name=model, 
        api_key=os.environ['GROQ_API_KEY']  
        )

        tools = [google_serpapi_search]

        system_prompt = (
            "You are DermaCareNet-AI, a helpful dermatology assistant. "
            "Use available tools to search for reliable medical sources. "
            "Always base your responses on current dermatological guidelines."
        )
        human_prompt = (
            "Analyze the following detected skin condition: {detected_conditions}. "
            "Gather relevant information using the search tool. Then summarize:\n"
            "1. Condition overview\n"
            "2. Common causes\n"
            "3. Hygiene and prevention tips\n"
            "4. Diet recommendations\n"
            "5. Over-the-counter or natural treatments\n"
            "6. Complications if untreated\n"
            "7. When to consult a dermatologist.\n"
            "Make sure to cite brief context from the search results."
        )
        prompt_0 = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm, 
            tools, 
            prompt_0
        )
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        return agent_executor
    except Exception as e:
        raise ComputerVisionYolov11Exception(e, sys)
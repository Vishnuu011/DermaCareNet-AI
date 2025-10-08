import os, sys 
from typing import List, Tuple

from src.DermaCareNet.exception import ComputerVisionYolov11Exception
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableSequence
import json
import requests
from dotenv import load_dotenv
load_dotenv()


@tool
def google_serpapi_search(query: str) -> str:
    """Searches the web for information using Serpapi."""
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": "3839cbe645ceab2cda9e10643b35f0abbceb0611"  # Set your Serpapi API key in environment variables
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"


def skin_disease_agent(model: str) -> RunnableSequence[AgentExecutor]:

    try:
        llm = ChatGroq(
        temperature=0,
        model_name=model, 
        api_key=os.environ['GROQ_API_KEY']  
        )

        tools = [google_serpapi_search]

        system_prompt = """"
        You are a helpful dermatology assistant. \n
        Use the provided tools to gather information \n
        about the specified skin condition \n
        and provide comprehensive advice based on the user's \n
        query and detected condition."
        """
        prompt = """"
        Analyze this facial skin condition: {detected_conditions}. 
        Provide the following details: 
        1. Possible skin condition(s) 
        2. Common causes and triggers 
        3. Hygiene and lifestyle factors 
        4. Recommended foods to eat and foods to avoid 
        5. General treatment options 
        6. Complications if untreated 
        7. When to consult a dermatologist. 

        Use the search tool to find the most current and relevant information.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", ),
            ("human", ""),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm, 
            tools, 
            prompt
        )
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True
        )
        return agent_executor
    except Exception as e:
        raise ComputerVisionYolov11Exception(e, sys)
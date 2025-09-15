import os
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import streamlit as st
from prompt_config import PROMPT_CONFIG
from typing import Any
from dotenv import load_dotenv
# Load env vars
load_dotenv()

def get_llm():
    return AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        max_tokens=1000,
    )

def get_agent(llm=None, system_prompt=None):
    if llm is None:
        llm = get_llm()

    if system_prompt is None:
        system_prompt = PROMPT_CONFIG["oncoally_system"]

    pubmed_tool = PubmedQueryRun()
    tools = [pubmed_tool]

    # ✅ Pass PromptTemplate instead of string
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    return agent

async def ask_question(question: str) -> Any:
    """
    Handles user question by sending it to the AI agent.

    Args:
        question (str): The question to ask the cancer assistant agent.

    Returns:
        Any: The agent's response (can be dict, str, etc. depending on agent design)
    """
    try:
        agent = get_agent()
        response = await agent.ainvoke({"messages": question})
        return response
    except Exception as e:
        # Log in production or send to error monitoring
        return {"error": str(e)}
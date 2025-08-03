import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from agent_utils import *
from prompt_config import PROMPT_CONFIG
import asyncio
# Load env vars
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="OncoAlley – Because Behind Every Question Is a Life",
    page_icon="🩺",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background: #f4f6fa;
    }
    .main {
        padding-top: 1rem;
    }
    .chat-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        max-height: 70vh;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .chat-bubble {
        margin-bottom: 1.25rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-bubble.user {
        justify-content: flex-end;
    }
    .chat-bubble.assistant {
        justify-content: flex-start;
    }
    .bubble {
        max-width: 70%;
        padding: 1rem 1.4rem;
        font-size: 16px;
        line-height: 1.5;
        border-radius: 18px;
    }
    .bubble.user {
        background: #4f46e5;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
        background: #e5e7eb;
        color: #1f2937;
        border-bottom-left-radius: 4px;
    }
    .avatar {
        width: 40px;
        height: 40px;
        margin: 0 10px;
        border-radius: 50%;
        background: #c7d2fe;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .input-row {
        display: flex;
        gap: 10px;
        margin-top: 1rem;
    }
    .input-row input {
        flex: 1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# --- Sidebar ---
st.sidebar.image("https://raw.githubusercontent.com/AJAnujsharma/ONCOALLY/main/ChatGPT%20Image%20Aug%201%2C%202025%2C%2009_18_22%20PM.png", width=180)
st.sidebar.markdown("### 🩺 OncoAlly")
st.sidebar.markdown("""
**Your AI-powered cancer support companion.**

- 👨‍💻 [Microsoft Global Hackathon 2025](https://innovationstudio.microsoft.com/hackathons/hackathon2025/project/96924)
- ⚡ *AI-Powered by Azure, PubMed & MCP Agentic Tech*
- 🔍 *Answers sourced from* **35M+ PubMed articles**

ℹ️ *Based on trusted data from the National Library of Medicine.*
""")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state["messages"] = []
    st.success("Chat cleared.")
if st.sidebar.button("🔁 Reset Agent"):
    if "agent" in st.session_state:
        del st.session_state["agent"]
    st.success("Agent has been reset.")

# --- Header ---
st.markdown("""
<div class="header">
    <h1>🩺 OncoAlley – Because Behind Every Question Is a Life</h1>
    <p>OncoAlly offers answers, not diagnoses. Whether you're seeking clarity on symptoms, treatments, or emotional care, we provide support rooted in trusted medical literature — always with compassion, never as a substitute for a doctor.</p>
</div>
""", unsafe_allow_html=True)

# --- Chat Container ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    role_class = "user" if msg["role"] == "user" else "assistant"
    icon = "🧑" if msg["role"] == "user" else "🩺"
    st.markdown(f"""
    <div class="chat-bubble {role_class}">
        <div class="avatar">{icon}</div>
        <div class="bubble {role_class}">{msg['content']}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Sample complex prompts
complex_questions = [
    "What are the treatment options for recurrent MSI-H endometrial cancer after progression on pembrolizumab?",
    "How does the presence of a TP53 mutation impact the prognosis and treatment strategy in high-grade serous ovarian cancer?",
    "Can you explain the role of BRCA1/2 mutations in selecting between PARP inhibitors and chemotherapy for metastatic breast cancer?",
    "What are the current recommendations for managing immune-related colitis in a patient receiving combination checkpoint inhibitors for metastatic melanoma?",
    "How does tumor mutational burden influence the choice of immunotherapy in non-small cell lung cancer with negative PD-L1 expression?"
]

st.markdown("### 🧠 Sample Expert Questions")
st.caption("Select a question from the dropdown to see how OncoAlly responds:")

# Dropdown selection
selected_question = st.selectbox("Choose a sample question:", ["-- Select --"] + complex_questions)

# Only show button if a real question is selected
if selected_question != "-- Select --":
    if st.button("Ask OncoAlly"):
        st.session_state["messages"].append({"role": "user", "content": selected_question})
        with st.spinner("OncoAlly is thinking..."):
            response = asyncio.run(ask_question(selected_question))
            answer = response["messages"][-1].content if "messages" in response and response["messages"] else "Sorry, I couldn't find an answer."
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

# --- Input UI ---
if st.session_state.get("clear_input"):
    st.session_state["user_input"] = ""
    st.session_state["clear_input"] = False
user_input = st.text_input("Ask OncoAlly your question...", placeholder="e.g. What are early signs of breast cancer?", key="user_input")
send, clear = st.columns([1, 1])


if send.button("Send", use_container_width=True):
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("OncoAlly is thinking..."):
            latest_question = st.session_state["messages"][-1]["content"]
            response = asyncio.run(ask_question(latest_question))
            # print(response)
            answer = response["messages"][-1].content if "messages" in response and response["messages"] else "Sorry, I couldn't find an answer."
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
if clear.button("Clear Input", use_container_width=True):
    st.session_state["clear_input"] = True
    st.rerun()

# --- Footer ---
st.markdown("""
<hr style="margin-top: 2rem;">
<div style="text-align:center; font-size:14px; color:#4f46e5;">
    <b>OncoAlly</b> &copy; 2025 – Crafted with ❤️ by the OncoAlly Core Engineering Team</b><br>
    Compassionate support backed by 35M+ biomedical citations.
</div>
""", unsafe_allow_html=True)

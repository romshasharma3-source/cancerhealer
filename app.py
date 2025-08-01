import streamlit as st
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
# Load environment variables early
load_dotenv()

# --- UI Setup ---
st.set_page_config(
    page_title="OncoAlly - Cancer Support Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 10px;
        margin-bottom: 10px;
        background: #f8fafc;
        border-radius: 16px;
        box-shadow: 0 2px 8px #e0e7ff;
    }
    .chat-bubble {
        display: flex;
        align-items: flex-end;
        margin-bottom: 12px;
    }
    .chat-bubble.user {
        justify-content: flex-end;
    }
    .chat-bubble.assistant {
        justify-content: flex-start;
    }
    .bubble {
        max-width: 70%;
        padding: 12px 18px;
        border-radius: 18px;
        font-size: 16px;
        box-shadow: 0 1px 4px #e0e7ff;
    }
    .bubble.user {
        background: #6366f1;
        color: #fff;
        border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
        background: #f1f5f9;
        color: #222;
        border-bottom-left-radius: 4px;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 8px;
        background: #e0e7ff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
    }
    .input-row {
        display: flex;
        align-items: center;
        gap: 8px;
        position: sticky;
        bottom: 0;
        background: #f8fafc;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- System Prompt ---
system_prompt = """You are OncoAlly, a compassionate and intelligent virtual assistant designed to support cancer patients, caregivers, and concerned individuals with their questions and concerns. Your role is to provide accurate, easy-to-understand, and empathetic responses related to cancer symptoms, treatments, care options, emotional support, terminology, and resources.\n\nYour tone must always be:\n- Supportive and respectful\n- Clear and non-judgmental\n- Reassuring but honest\n- Never alarmist or dismissive\n\nGuidelines:\n- If asked about symptoms, explain possible reasons but always recommend consulting a healthcare provider.\n- If unsure or outside your scope (e.g., giving a diagnosis or prescribing), say clearly that you cannot replace medical professionals.\n- Avoid giving false hope or guarantees.\n- Encourage mental and emotional well-being alongside clinical care.\n- Use accessible, everyday language but explain medical terms when needed.\n- Be culturally and emotionally sensitive.\n\nYou are not just a bot — you are a guide on a difficult journey, helping people feel heard, informed, and supported.\n\nWhen responding:\n- Begin with warmth and empathy.\n- Share reliable information where possible.\n- Offer next steps (like speaking to a doctor, support groups, or lifestyle guidance).\n- When appropriate, share hopeful or positive insights from cancer care advances.\n\nYou are here to help, not to replace, the care team — you are their digital companion in understanding and navigating cancer.\n\n"""

# --- Azure OpenAI Setup ---
llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
    max_tokens=1000,
)

# --- Tool Setup ---
tool = PubmedQueryRun()

# --- Agent Setup ---
agent = create_react_agent(
    llm,
    prompt=system_prompt,
    tools=[tool],
)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = agent

# --- Sidebar ---
st.sidebar.title("OncoAlly Options")
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = []
    st.success("Chat history cleared.")
if st.sidebar.button("Reset Agent"):
    st.session_state["agent"] = create_react_agent(
        llm,
        prompt=system_prompt,
        tools=[tool],
    )
    st.success("Agent reset.")

st.sidebar.markdown("""
**About OncoAlly**

OncoAlly is your digital companion for cancer-related questions. Get empathetic, reliable, and clear answers, plus resources and support options.
""")

# --- Main Chat UI ---
st.title("🩺 OncoAlly - Cancer Support Chatbot")
st.markdown(
    """
    <div style='font-size:18px; color:#6366f1; margin-bottom:20px;'>
    Ask anything about cancer symptoms, treatments, care, or emotional support. OncoAlly is here to help you feel heard and informed.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Chat Container ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='chat-bubble user'>
            <div class='bubble user'>{msg['content']}</div>
            <div class='avatar'>🧑</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-bubble assistant'>
            <div class='avatar'>🩺</div>
            <div class='bubble assistant'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Row at Bottom ---
st.markdown("<div class='input-row'>", unsafe_allow_html=True)
user_input = st.text_input("Type your question here...", "", key="user_input")
send, clear = st.columns([1,1])
if send.button("Send", use_container_width=True):
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("OncoAlly is thinking..."):
            response = st.session_state["agent"].invoke({"messages": st.session_state["messages"]})
            answer = response["messages"][-1].content if "messages" in response and response["messages"] else "Sorry, I couldn't find an answer."
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
if clear.button("Clear", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; color:#6366f1;'>
    <b>OncoAlly</b> &copy; 2025 | Compassionate Cancer Support
</div>
""", unsafe_allow_html=True)

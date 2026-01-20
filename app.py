import streamlit as st
import random
import uuid
from chat_logic import get_response
from facts import FACTS

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Eleazar Phoenix AI",
    page_icon="🦅",
    layout="wide" 
)

# --- MAJESTIC DARK THEME CSS ---
st.markdown("""
    <style>
    /* Background: Deep Charcoal to Navy Gradient */
    .stApp {
        background: linear-gradient(180deg, #1a1a1a 0%, #0d1117 100%);
        color: #e0e0e0;
    }
    
    /* Ensure content doesn't get hidden behind the fixed footer */
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Header & Titles */
    .main-title {
        font-size: 3.2rem !important;
        font-weight: 800;
        color: #E5B4B2; /* Rose Gold / Majestic Pink-Gold */
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 2px;
    }
    .description {
        font-size: 1.2rem;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 30px;
        font-style: italic;
    }

    /* Sidebar Separation */
    [data-testid="stSidebar"] {
        background-color: #111418 !important;
        border-right: 2px solid #E5B4B2;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        margin-top: 10px;
    }

    /* Suggestion Chips */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #E5B4B2;
        background-color: transparent;
        color: #E5B4B2;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #E5B4B2;
        color: #1a1a1a;
    }

    /* Footer - Enhanced Visibility */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0d1117;
        color: #E5B4B2;
        text-align: center;
        padding: 15px;
        font-size: 0.9rem;
        border-top: 1px solid #E5B4B2;
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #E5B4B2;'>🎂 The Legacy</h2>", unsafe_allow_html=True)
    st.write("Explore the life of Mr. Eleazar Olumuyiwa Ogunmilade.")
    st.divider()
    
    # Fact Box
    st.markdown("<div style='padding:10px; border:1px solid #30363d; border-radius:10px;'>", unsafe_allow_html=True)
    st.subheader("📌 Life Fact")
    if "current_fact" not in st.session_state:
        st.session_state.current_fact = random.choice(FACTS)
    st.write(st.session_state.current_fact)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("✨ Show Another Fact"):
        st.session_state.current_fact = random.choice(FACTS)
        st.rerun()

# --- MAIN LANDING ---
st.markdown('<h1 class="main-title">ELEAZAR PHOENIX AI 🎂</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Celebrating the man, the myth, the legend Eleazar Olumuyiwa Ogunmilade</p>', unsafe_allow_html=True)

# --- SESSION INITIALIZATION (FIXED WITH UUID) ---
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Eleazar Phoenix AI. I am here to tell you about the incredible journey and character of Mr. Eleazar. What would you like to know about him?"}
    ]

# --- CLICKABLE QUESTIONS ---
st.markdown("### 🔍 Discover more about him:")
c1, c2, c3 = st.columns(3)
if c1.button("Who is Mr. Eleazar?"): st.session_state.pending_input = "Who is Mr. Eleazar?"
if c2.button("Tell me about his career"): st.session_state.pending_input = "Tell me about Mr. Eleazar's career and achievements."
if c3.button("His impact on others"): st.session_state.pending_input = "What is Mr. Eleazar's impact on those around him?"

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT LOGIC ---
user_input = st.chat_input("Ask about his legacy...")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = get_response(user_input, session_id=st.session_state.user_session_id)
            full_response = response.content if hasattr(response, 'content') else str(response)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")

# --- FOOTER (Developed by Me) ---
st.markdown(
    '<div class="footer">Developed by Olaleye Faithfulness Ibukun | © 2026 Legacy Tribute</div>', 
    unsafe_allow_html=True
)
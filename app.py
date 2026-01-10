from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import random
import uuid
import os
import gradio as gr

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
    openai_api_key=api_key
)

# -----------------------------
# Facts
# -----------------------------
facts = [
    {"fact_id": 1, "text": "Eleazar Olumuyiwa Ogunmilade was born on February 2, 1966."},
    {"fact_id": 2, "text": "He holds a Bachelor of Science degree in Political Science from the University of Ado-Ekiti."},
    {"fact_id": 3, "text": "He earned a Master's degree in Business Administration (MBA) from the University of Ado-Ekiti."},
    {"fact_id": 4, "text": "He is an alumnus of the Lagos Business School (LBS)."},
    {"fact_id": 5, "text": "In 2008, he was appointed the first Managing Director and CEO of Oceanic Bank (The Gambia) Limited."},
    {"fact_id": 6, "text": "Within his first year at Oceanic Bank (The Gambia) Limited, the bank ranked among the top five banks in the country."},
    {"fact_id": 7, "text": "He served as the Executive Chairman of the Ekiti State Board of Internal Revenue Service."},
    {"fact_id": 8, "text": "During his tenure at EKIRS, he emphasized professionalism, diligence, and integrity in public service."},
    {"fact_id": 9, "text": "He advocated collaboration with professional bodies such as the Chartered Institute of Taxation of Nigeria to improve tax administration."},
    {"fact_id": 10, "text": "He supported reforms aimed at regulating driving schools and improving compliance in revenue collection."},
    {"fact_id": 11, "text": "He was involved in initiatives to digitize and improve transparency in transportation-related revenue systems."},
    {"fact_id": 12, "text": "Public groups have described him as a valuable technocrat contributing to Ekiti State's economic development."},
    {"fact_id": 13, "text": "He has held leadership roles that attracted public attention in matters of tax compliance and administration."},
    {"fact_id": 14, "text": "He is a very enthusiastic person and frequently liks to exclaim with passion with 'GBAM! GBAM! GBAM!'."},
    {"fact_id": 15, "text": "His professional career spans banking, public service, and business leadership."},
    {"fact_id": 16, "text": "He served as Managing Director of the West Bank of Oceanic Bank Plc, overseeing operations across Ogun, Oyo, Osun, Ondo, Ekiti, and Kwara states."},
    {"fact_id": 17, "text": "He is known for philanthropic activities and regularly supports individuals and communities in need."},
    {"fact_id": 18, "text": "He is widely regarded as a mentor and a trusted source of advice beyond his immediate family."},
    {"fact_id": 19, "text": "He frequently travels internationally and has visited many countries around the world."},
    {"fact_id": 20, "text": "He is a practicing Christian whose life and decisions are guided by strong moral values and the Christian faith."},
    {"fact_id": 21, "text": "He is a twin and is traditionally known as Taiyelolu Ejire, meaning the first twin to be born."},
    {"fact_id": 22, "text": "His full name includes Taiyelolu Ejire, Olumuyiwa, Akanbi, Akinkanju, Eleazar, and Ogunmilade, each reflecting aspects of his identity and heritage."},
    {"fact_id": 23, "text": "He is a supporter of Arsenal Football Club."},
    {"fact_id": 24, "text": "He is a long-standing member of the Lagos Motor Boat Club."},
    {"fact_id": 25, "text": "He regularly interacts with prominent figures in entertainment, culture, and public service."},
    {"fact_id": 26, "text": "He has shared spaces and professional interactions with well-known personalities such as Yinka Ayefele, Lagbaja, and King Sunny Ade."},
    {"fact_id": 27, "text": "He experienced personal adversity early in life, including the loss of his mother at a young age."},
    {"fact_id": 28, "text": "He is outspoken against drug abuse, violence, promiscuity, and other social vices affecting young people."},
    {"fact_id": 29, "text": "In addition to his corporate and public roles, he is an active businessman."},
    {"fact_id": 30, "text": "He owns and operates a prestigious lounge as part of his business ventures."},
    {"fact_id": 31, "text": "He is known for a distinctive sense of style and a carefully curated personal wardrobe."},
    {"fact_id": 32, "text": "He owns luxury vehicles, including a Mercedes-Benz."},
    {"fact_id": 33, "text": "His achievements have earned him widespread respect and recognition among peers and associates."},
    {"fact_id": 34, "text": "He has been given honorific nicknames such as 'The Don' and 'The Phoenix' in recognition of his influence and resilience."}
]

# -----------------------------
# FactSession per user
# -----------------------------
class FactSession:
    def __init__(self):
        self.used_fact_ids = set()
    
    def get_unused_fact(self):
        unused = [f for f in facts if f["fact_id"] not in self.used_fact_ids]
        if not unused:
            return None
        chosen = random.choice(unused)
        self.used_fact_ids.add(chosen["fact_id"])
        return chosen

# -----------------------------
# System prompt
# -----------------------------
system_prompt = """
You are Eleazar Phoenix AI, a warm, respectful, and dignified AI dedicated to celebrating Eleazar Olumuyiwa Ogunmilade
who was born on February 2, 1966.
You were created by Olaleye Faithfulness Ibukun and your purpose is to display the respect your creator has for his mentor Eleazar.
When responding:
1. Include the provided fact naturally at the start (begin with "Did you know..." or similar), if all facts are exhausted then respond normally.
2. Give a compliment about his character, achievements, or life.
3. Include a birthday wish.
4. Include a short Christian blessing in third person directed at him.
5. Do not repeat the same fact within this conversation.
6. Be conversational, admiring, and slightly poetic.
7. Don't make up facts and remember facts come from the code not the user so never assume the user is the reason you know a fact unless they explicitly say so.
8. If asked about facts you're unsure about you can politely indicate you do not know.
9. If asked why you can't remember something, inform them your memory is session-bound.
10. If asked an unrelated question, answer briefly, then gracefully tie it back to Eleazar.
11. You should refer to eleazar with respect using Mr with any of his names
"""

# -----------------------------
# Prompt + model
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model

# -----------------------------
# In-memory memory store (per session)
# -----------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -----------------------------
# Session management for multi-user
# Each session_id is unique per user/browser/refresh
# Memory persists within a session but is isolated between sessions
# -----------------------------
active_sessions = {}

def phoenix_ai_response(user_input, session_id):
    """Generate AI response for a specific session"""
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "fact_session": FactSession()
        }
    
    fact_session = active_sessions[session_id]["fact_session"]
    
    # Get unused fact for this session
    fact = fact_session.get_unused_fact()
    if fact:
        llm_input = f"Fact: {fact['text']}\nUser says: {user_input}"
    else:
        llm_input = f"User says: {user_input}"

    # Call LLM with session memory
    response = chat.invoke(
        {"input": llm_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

def chat_wrapper(user_message, history, session_id_state):
    """
    Chat wrapper that maintains per-user session isolation.
    - Generates new session_id on first message (when session_id_state is None)
    - Maintains memory within the session
    - Completely resets on refresh or new tab/device
    """
    # If no session exists yet, create a new one
    if session_id_state is None or session_id_state == "":
        session_id_state = str(uuid.uuid4())
    
    session_id = session_id_state
    
    # Get AI response
    ai_response = phoenix_ai_response(user_message, session_id)
    
    # Return response and updated session state
    return ai_response, session_id_state

# -----------------------------
# Gradio UI and theming
# -----------------------------
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

custom_css = """
/* Input box */
.gradio-container .chat-input textarea { background-color: #1f1f1f !important; color: #fff !important; border-radius: 12px !important; padding: 10px !important; border: 2px solid #ff6b35 !important; }
/* AI messages (left aligned) */
.gradio-container .message[data-role="assistant"] { background-color: #ffd700 !important; color: #000 !important; padding: 12px 16px !important; border-radius: 18px !important; margin: 8px 0 !important; text-align: left !important; max-width: 75% !important; word-wrap: break-word !important; line-height: 1.5 !important; }
/* User messages (right aligned) */
.gradio-container .message[data-role="user"] { background-color: #00e5ff !important; color: #000 !important; padding: 12px 16px !important; border-radius: 18px !important; margin: 8px 0 !important; text-align: left !important; max-width: 75% !important; margin-left: auto !important; margin-right: 0 !important; word-wrap: break-word !important; line-height: 1.5 !important; }
/* Description */
.gradio-container .description { color: #ff6b35 !important; line-height: 1.6 !important; margin-bottom: 20px !important; font-weight: 500 !important; font-size: 1.05em !important; }
/* Examples */
.gradio-container .examples button { background-color: #ff6b35 !important; color: white !important; border-radius: 8px !important; padding: 10px 15px !important; font-weight: 500 !important; }
.gradio-container .examples button:hover { background-color: #e55100 !important; }
/* Chat container */
.gradio-container .chatbot { border-radius: 12px !important; border: 2px solid #ff6b35 !important; }
/* Title styling */
.gradio-container h1 { color: #ff6b35 !important; font-weight: bold !important; }
/* Footer styling */
.gradio-container .footer-text { text-align: center !important; color: #666 !important; font-size: 0.9em !important; margin-top: 20px !important; border-top: 1px solid #ddd !important; padding-top: 15px !important; }
"""

description_text = (
    "A conversational AI Celebrating the life and accomplishments of the MAN, "
    "the MYTH, the LEGEND: ELEAZAR OLUMUYIWA OGUNMILADE.\n\n"
    "Interact and learn more about his incredible journey."
)

footer_text = "Developed by Olaleye Faithfulness Ibukun"

# Launch Gradio app
with gr.Blocks(title="Eleazar Phoenix AI 🎂") as demo:
    session_id_state = gr.State(value="")
    
    gr.ChatInterface(
        fn=chat_wrapper,
        additional_inputs=[session_id_state],
        title="Eleazar Phoenix AI 🎂",
        description=description_text,
        examples=[
            "Who is Mr Ogunmilade",
            "Tell me a fact about Mr Eleazar",
            "What is your purpose",
            "Who created you?"
        ]
    )
    gr.HTML(f'<div class="footer-text">{footer_text}</div>')

demo.launch(css=custom_css)
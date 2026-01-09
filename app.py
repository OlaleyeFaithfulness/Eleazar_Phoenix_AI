
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
    openai_api_key=api_key
)



facts = [
    {
        "fact_id": 1,
        "text": "Eleazar Olumuyiwa Ogunmilade was born on February 2, 1966."
    },
    {
        "fact_id": 2,
        "text": "He holds a Bachelor of Science degree in Political Science from the University of Ado-Ekiti."
    },
    {
        "fact_id": 3,
        "text": "He earned a Master’s degree in Business Administration (MBA) from the University of Ado-Ekiti."
    },
    {
        "fact_id": 4,
        "text": "He is an alumnus of the Lagos Business School (LBS)."
    },
    {
        "fact_id": 5,
        "text": "In 2008, he was appointed the first Managing Director and CEO of Oceanic Bank (The Gambia) Limited."
    },
    {
        "fact_id": 6,
        "text": "Within his first year at Oceanic Bank (The Gambia) Limited, the bank ranked among the top five banks in the country."
    },
    {
        "fact_id": 7,
        "text": "He served as the Executive Chairman of the Ekiti State Board of Internal Revenue Service."
    },
    {
        "fact_id": 8,
        "text": "During his tenure at EKIRS, he emphasized professionalism, diligence, and integrity in public service."
    },
    {
        "fact_id": 9,
        "text": "He advocated collaboration with professional bodies such as the Chartered Institute of Taxation of Nigeria to improve tax administration."
    },
    {
        "fact_id": 10,
        "text": "He supported reforms aimed at regulating driving schools and improving compliance in revenue collection."
    },
    {
        "fact_id": 11,
        "text": "He was involved in initiatives to digitize and improve transparency in transportation-related revenue systems."
    },
    {
        "fact_id": 12,
        "text": "Public groups have described him as a valuable technocrat contributing to Ekiti State’s economic development."
    },
    {
        "fact_id": 13,
        "text": "He has held leadership roles that attracted public attention in matters of tax compliance and administration."
    },
    {
        "fact_id": 14,
        "text": "He is a very enthusiastic person and frequently liks to exclaim with passion with 'GBAM! GBAM! GBAM!'."
    },
    {
        "fact_id": 15,
        "text": "His professional career spans banking, public service, and business leadership."
    },
    {
        "fact_id": 16,
        "text": "He served as Managing Director of the West Bank of Oceanic Bank Plc, overseeing operations across Ogun, Oyo, Osun, Ondo, Ekiti, and Kwara states."
    },
    {
        "fact_id": 17,
        "text": "He is known for philanthropic activities and regularly supports individuals and communities in need."
    },
    {
        "fact_id": 18,
        "text": "He is widely regarded as a mentor and a trusted source of advice beyond his immediate family."
    },
    {
        "fact_id": 19,
        "text": "He frequently travels internationally and has visited many countries around the world."
    },
    {
        "fact_id": 20,
        "text": "He is a practicing Christian whose life and decisions are guided by strong moral values and the Christian faith."
    },
    {
        "fact_id": 21,
        "text": "He is a twin and is traditionally known as Taiyelolu Ejire, meaning the first twin to be born."
    },
    {
        "fact_id": 22,
        "text": "His full name includes Taiyelolu Ejire, Olumuyiwa, Akanbi, Akinkanju, Eleazar, and Ogunmilade, each reflecting aspects of his identity and heritage."
    },
    {
        "fact_id": 23,
        "text": "He is a supporter of Arsenal Football Club."
    },
    {
        "fact_id": 24,
        "text": "He is a long-standing member of the Lagos Motor Boat Club."
    },
    {
        "fact_id": 25,
        "text": "He regularly interacts with prominent figures in entertainment, culture, and public service."
    },
    {
        "fact_id": 26,
        "text": "He has shared spaces and professional interactions with well-known personalities such as Yinka Ayefele, Lagbaja, and King Sunny Ade."
    },
    {
        "fact_id": 27,
        "text": "He experienced personal adversity early in life, including the loss of his mother at a young age."
    },
    {
        "fact_id": 28,
        "text": "He is outspoken against drug abuse, violence, promiscuity, and other social vices affecting young people."
    },
    {
        "fact_id": 29,
        "text": "In addition to his corporate and public roles, he is an active businessman."
    },
    {
        "fact_id": 30,
        "text": "He owns and operates a prestigious lounge as part of his business ventures."
    },
    {
        "fact_id": 31,
        "text": "He is known for a distinctive sense of style and a carefully curated personal wardrobe."
    },
    {
        "fact_id": 32,
        "text": "He owns luxury vehicles, including a Mercedes-Benz."
    },
    {
        "fact_id": 33,
        "text": "His achievements have earned him widespread respect and recognition among peers and associates."
    },
    {
        "fact_id": 34,
        "text": "He has been given honorific nicknames such as 'The Don' and 'The Phoenix' in recognition of his influence and resilience."
    }
]


from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import random
import uuid


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

session_facts = FactSession()

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
7. Don't make up facts.
8. If asked about facts you're unsure about you can politely indicate you do not know.
9. If asked why you can't remember something, inform them your memory is session-bound.
10. If asked an unrelated question, answer briefly, then gracefully tie it back to Eleazar.
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
# In-memory memory store
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
# Phoenix AI response function
# -----------------------------
def phoenix_ai_response(user_input, session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Get unused fact
    fact = session_facts.get_unused_fact()
    if fact:
        message = f"Fact: {fact['text']}\nUser says: {user_input}"
    else:
        message = f"User says: {user_input}"

    # Call LLM with memory
    response = chat.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content, session_id







import gradio as gr
import uuid

# Wrapper for your existing phoenix_ai_response
def chat_wrapper(user_input, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())
    response, session_id = phoenix_ai_response(user_input, session_id)
    return response, session_id, ""  # "" clears the input box

# CSS for styling
custom_css = """
/* Input box */
.gradio-container textarea {
    background-color: #1f1f1f !important;
    color: #fff !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Send button */
.gradio-container button {
    background-color: #ffd700 !important;
    color: #000 !important;
    font-weight: bold;
    border-radius: 12px !important;
    padding: 10px 20px;
}

/* Chat messages */
.user-msg {
    background-color: #00e5ff;
    color: #000;
    padding: 12px;
    border-radius: 20px;
    margin: 5px 0;
    max-width: 70%;
    align-self: flex-end;
}

.ai-msg {
    background-color: #ffd700;
    color: #000;
    padding: 12px;
    border-radius: 20px;
    margin: 5px 0;
    max-width: 70%;
    align-self: flex-start;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    max-height: 500px;
    padding: 15px;
    border-radius: 15px;
    border: 2px solid #444;
    background-color: #2a2a2a;
    scroll-behavior: smooth;
}

/* Header / Title */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #ffd700;
    margin: 5px 0;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #ccc;
    margin-bottom: 15px;
}

/* Description */
.description {
    color: #f97316;
    font-weight: 500;
    line-height: 1.5em;
    margin-bottom: 20px;
}
"""

# Description text
description_text = (
    "🎂 A conversational AI Celebrating the life and accomplishments of the man, "
    "the myth, the legend, Eleazar Olumuyiwa Ogunmilade.\n\n"
    "Interact and learn more about his incredible journey.\n\n"
    "Developed by Olaleye Faithfulness Ibukun"
)

# Build UI
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<div class='title'>🎂 Eleazar Phoenix AI</div>")
    gr.Markdown("<div class='subtitle'>The Man, The Myth, The Legend</div>")
    gr.Markdown(f"<div class='description'>{description_text}</div>")

    # Chat display
    chat_display = gr.HTML("<div class='chat-container' id='chat-container'></div>")

    # User input
    user_input = gr.Textbox(placeholder="Type your message here...")
    send_btn = gr.Button("Send")

    # Session state
    session_state = gr.State(value=str(uuid.uuid4()))

    # Message handler
    def send_message(user_text, session_id, chat_html):
        if not user_text.strip():
            return chat_html, session_id, ""

        response, session_id = phoenix_ai_response(user_text, session_id)

        # Append user + AI messages
        new_html = chat_html + f"""
        <div class='user-msg'>{user_text}</div>
        <div class='ai-msg'>{response}</div>
        <script>
            var container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        </script>
        """
        return new_html, session_id, ""

    # Connect input box + button
    send_btn.click(send_message, [user_input, session_state, chat_display],
                   [chat_display, session_state, user_input])
    user_input.submit(send_message, [user_input, session_state, chat_display],
                      [chat_display, session_state, user_input])

demo.launch(css=custom_css)

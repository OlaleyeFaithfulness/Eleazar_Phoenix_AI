
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

# Your existing backend
def chat_wrapper(message, history):
    if not hasattr(chat_wrapper, "session_id"):
        chat_wrapper.session_id = str(uuid.uuid4())
    bot_response, _ = phoenix_ai_response(message, chat_wrapper.session_id)
    return bot_response

# CSS for message styling
custom_css = """
.user {
    background: #00e5ff;
    color: #000;
    border-radius: 12px;
    padding: 8px 12px;
    margin: 2px 0;
    align-self: flex-end;
    max-width: 75%;
}
.bot {
    background: #ffd700;
    color: #000;
    border-radius: 12px;
    padding: 8px 12px;
    margin: 2px 0;
    align-self: flex-start;
    max-width: 75%;
}
"""

description_text = """
🎂 **A conversational AI celebrating the life and accomplishments of  
the man, the myth, the legend — Eleazar Olumuyiwa Ogunmilade.**

Interact and learn more about his incredible journey.
"""

footer_text = "Developed by Olaleye Faithfulness Ibukun"

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎂 Eleazar Phoenix AI")
    gr.Markdown(description_text)

    chatbot = gr.Chatbot(elem_id="chatbot-window").style(height=450)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            interactive=True,
        )
        send_btn = gr.Button("Send")

    # suggested prompts
    with gr.Accordion("Suggested Prompts", open=True):
        gr.Markdown(
            "- Tell me a fact about Mr Eleazar\n"
            "- When is Mr Ogunmilade’s birthday?\n"
            "- Who created you?\n"
            "- Give me a blessing for him\n"
            "- What’s something inspiring about his life?"
        )

    gr.Markdown(f"---\n*{footer_text}*")

    def handle_send(user_text, chat_history):
        if not user_text.strip():
            return chat_history, ""
        # send to backend
        ai_text = chat_wrapper(user_text, chat_history)
        # append to chat history
        chat_history = chat_history + [(user_text, ai_text)]
        return chat_history, ""

    send_btn.click(handle_send, [msg, chatbot], [chatbot, msg])
    msg.submit(handle_send, [msg, chatbot], [chatbot, msg])

demo.launch()

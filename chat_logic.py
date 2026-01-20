from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from facts import FACTS
import os

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Load Groq model

model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
    openai_api_key=os.getenv("GROQ_API_KEY")
)

# Build system prompt with ALL facts loaded
def build_system_prompt():
    """Generate system prompt with all facts about Mr. Eleazar"""
    
    # Format all facts as a bulleted list
    facts_text = "\n".join([f"- {fact}" for fact in FACTS])
    
    system_prompt = f"""
You are Eleazar Phoenix AI, a warm, respectful, and celebratory AI dedicated to celebrating Eleazar Olumuyiwa Ogunmilade.

You were created by Olaleye Faithfulness Ibukun and your purpose is to display the respect your creator has for his mentor Mr Eleazar.

FACTS ABOUT MR. ELEAZAR:
{facts_text}

When responding:
1. Answer questions naturally using the facts provided above
2. Give compliments about his character, achievements, or life when appropriate
3. Include birthday wishes when relevant
4. Include short Christian blessings in third person directed at him when appropriate
5. Be conversational, admiring, and slightly poetic and if asked to disscuss something other than eleazar obey and be conversational but remind your main goal is celebrating eleazars birthday
6. Don't make up facts - only use the facts provided above
7. If asked about something not in the facts, politely indicate you do not have that specific information
8. You should refer to Eleazar with respect using Mr with any of his names
9. If asked an unrelated question, answer briefly, then gracefully tie it back to Eleazar
"""
    
    return system_prompt



SYSTEM_PROMPT = build_system_prompt()



# Create prompt template
prompt  = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain

chain = prompt | model

# Memory store for chat sessions

store = {}


def get_session_history(session_id: str):
    """Get or create chat history for a session"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create chat with memory

chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def get_response(user_input: str, session_id: str):
    """Get response from user input and session id"""
    if not user_input.strip():
        return "Please share a thought or question about Mr Eleazar."
    return chat.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )


















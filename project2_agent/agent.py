from tools import ALL_TOOLS
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai.chat_models import ChatOpenAI

SYSTEM_PROMPT = (
    "Use calculator for every calculation, use get_student_grades for marks, "
    "remember earlier messages, answer briefly."
)

def build_agent(model):

    memory = InMemorySaver()
    agent = create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )
    
    return agent
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from agent import build_agent

load_dotenv(find_dotenv())



def main():

    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL_NAME"), base_url=os.getenv("INCEPTION_BASE_URL"),
                    api_key=os.getenv("INCEPTION_API_KEY"), temperature=0)
    # agent creation
    agent = build_agent(llm)

    #thread id creation
    thread_id = input("Enter thread ID (default 'default'): ").strip()
    if not thread_id:
        thread_id = "default"

    show_tools_input = input("Show tool calls? y/n: ").strip().lower()
    verbose = show_tools_input == 'y'
    config = {"configurable": {"thread_id": thread_id}}

    print("\n--- Chat Started (Type 'exit' or 'quit' to end) ---\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        if not user_input:
            continue

        input_messages = {"messages": [("user", user_input)]}

        try:
            for chunk in agent.stream(input_messages, config=config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    messages = node_update.get("messages", [])
                    for message in messages:
                        # Print agent response messages
                        if node_name == "agent" and message.content:
                            print(f"\nAI: {message.content}")

                        # In verbose mode, print tool calls and tool results
                        if verbose:
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    print(f"\n[Tool Call]: {tool_call['name']}({tool_call['args']})")
                            if node_name == "tools":
                                print(f"[Tool Result]: {message.content}")
        except Exception as e:
            print("\n[!] The AI provider is currently overloaded or rate-limited.")
            print("[!] Please wait a few seconds and try your question again.")

if __name__ == "__main__":
    main()
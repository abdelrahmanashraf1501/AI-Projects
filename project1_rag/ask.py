from config import CHAT_MODEL
from rag_chain import ask_question

def main():
    print("--- Welcome to the RAG System (Type 'exit' to quit) ---")
    
    while True:

        user_question = input("\nEnter your question: ").strip()
        if user_question.lower() == 'exit':
            print("Exiting...")
            break

        if not user_question:
            continue
            
        user_search_type = input("Enter search type ('mmr' or 'similarity') [Press Enter for mmr]: ").strip().lower()
        if not user_search_type:
            user_search_type = "mmr"
            
        print("\nThinking...\n")
        response = ask_question(question=user_question, search_type=user_search_type)
        
        print(f"Answer:\n{response}")

if __name__ == "__main__":
    main()
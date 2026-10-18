from config import TOP_K, DATA_DIR, DB_DIR, CHAT_MODEL, EMBED_MODEL
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
import os


system_prompt = "\n".join(
    [
        "You are a helpful assistant."
        "Use the following pieces of context to answer the question at the end."
        "If you don't know and have the answer, just say that you don't know, don't try to make up an answer."
        "CRITICAL INSTRUCTION: You MUST cite the source and page number for every claim you make.",
        "Format your citations inline like this: (Source: filename.pdf/.txt, Page: X/Y/Z)."
    ]
)

user_prompt = "\n".join(
    [
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Helpful Answer:"
    ]
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", user_prompt)
    ]
)

def get_retriever(vectorstore, search_type: str) -> BaseRetriever:
    
    try:
        if search_type == "mmr":
            return vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": TOP_K, "fetch_k": 10, "lambda_mult": 0.7})
        elif search_type == "similarity":
            return vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": TOP_K})
    except Exception as e:
        print(f"you have entered an invalid search type: {search_type}. Please choose either 'similarity' or 'mmr'.")

def format_docs_for_prompt(docs: list[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append(f"Context {i+1}: [source: {os.path.basename(doc.metadata['source'])}, page: {doc.metadata['page_label']}]:\n{doc.page_content}\n")
    return "\n".join(formatted_docs)

def build_rag_chain(vectorstore, llm, search_type: str):
    retriever_chain = get_retriever(vectorstore, search_type) | format_docs_for_prompt
    full_chain = {"context": retriever_chain, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()
    return full_chain


_vectorstore = Chroma(
    persist_directory=DB_DIR, 
    embedding_function=EMBED_MODEL
)

def ask_question(question: str, search_type: str = "mmr") -> str:
    """Builds the chain with the requested search type and runs the query."""
    
    # Validate the input to prevent crashes
    valid_types = ["mmr", "similarity"]
    if search_type not in valid_types:
        print(f"Warning: '{search_type}' is invalid. Defaulting to 'mmr'.")
        search_type = "mmr"
        
    chain = build_rag_chain(_vectorstore, CHAT_MODEL, search_type)
    return chain.invoke(question)
from config import DATA_DIR,DB_DIR,CHUNK_SIZE,CHUNK_OVERLAP,EMBED_MODEL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os
import shutil

def load_documents(file_path: str) -> list[Document]:
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

def create_vectorstore(documents: list[Document], persist_directory: str = DB_DIR) -> Chroma:

    if persist_directory and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_documents(documents, EMBED_MODEL, persist_directory=persist_directory)
    return vectorstore


if __name__ == "__main__":
    if not os.listdir(DATA_DIR):
        raise ValueError(f"No files found in {DATA_DIR}. Please add files to ingest.")
    else:
        all_split_docs = []

        for file_name in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file_name)
            documents = load_documents(file_path)
            split_docs = split_documents(documents)
            all_split_docs.extend(split_docs)
            
        all_split_docs = [doc for doc in all_split_docs if doc.page_content.strip()]
        create_vectorstore(all_split_docs)

        print(f"Vectorstore created and persisted in {DB_DIR}.")
        print(f"Number of files ingested: {len(os.listdir(DATA_DIR))} --> Number of chunks created: {len(all_split_docs)}")
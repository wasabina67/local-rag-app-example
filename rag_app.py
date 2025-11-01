import os
from typing import List

import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
)

# Index, Data paths
INDEX_DIR = "./index"
DATA_DIR = "./data"

# Ollama model configuration
LLM_MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


def load_documents() -> List[Document]:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        st.warning("'{DATA_DIR}' を新規作成しました。")
        return []

    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        return documents
    except Exception as e:
        st.error(e)
        return []


def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "何か気になることはありますか？"}
        ]


def main():
    st.set_page_config(
        page_title="Local rag app",
        page_icon="💬",
        layout="wide",
    )
    st.title("Local rag app")

    initialize_chat_history()

    with st.spinner("ドキュメントを読み込んでいます..."):
        documents = load_documents()


if __name__ == "__main__":
    main()

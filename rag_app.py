import os
from typing import List

import streamlit as st
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    load_index_from_storage,  # type: ignore
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore
from llama_index.llms.ollama import Ollama  # type: ignore

# Index, Data paths
INDEX_DIR = "./index"
DATA_DIR = "./data"

# Ollama model configuration
LLM_MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


def create_or_load_index(documents: List[Document], embed_model: OllamaEmbedding):
    if os.path.exists(INDEX_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            index = load_index_from_storage(storage_context)  # type: ignore
            assert isinstance(index, VectorStoreIndex)
            st.success("既存のインデックスを読み込みました。")
            return index
        except Exception as e:
            st.warning(e)
            st.info("新しいインデックスを作成します。")

    if not documents:
        st.error("インデックスを作成するドキュメントがありません。")
        return None

    try:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )
        os.makedirs(INDEX_DIR, exist_ok=True)
        index.storage_context.persist(persist_dir=INDEX_DIR)  # type: ignore
        st.success(
            f"{len(documents)} 個のドキュメントから新しいインデックスを作成しました。"
        )
        return index
    except Exception as e:
        st.error(e)
        return None


def setup_ollama_models():
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=360.0,
    )
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Set global settings for llama-index
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model


def load_documents() -> List[Document]:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        st.warning(f"{DATA_DIR} を新規作成しました。")
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
        page_icon=None,
        layout="wide",
    )
    st.title("Local rag app")

    initialize_chat_history()

    with st.spinner("ドキュメントを読み込んでいます..."):
        documents = load_documents()

    with st.spinner("Ollamaモデルを設定しています..."):
        llm, embed_model = setup_ollama_models()

    with st.spinner("インデックスを準備しています..."):
        index = create_or_load_index(documents, embed_model)

    if index is None:
        st.error(
            "インデックスが利用できません。ドキュメントを追加して再度お試しください。"
        )
        return

    query_engine = index.as_query_engine(llm=llm)  # type: ignore

    st.write("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("質問を入力してください"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("考えています..."):
                query_with_instruction = (
                    f"{prompt}\nこの質問に日本語で回答してください。"
                )
                response = query_engine.query(query_with_instruction)
                response_text = str(response)
                st.write(response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


if __name__ == "__main__":
    main()

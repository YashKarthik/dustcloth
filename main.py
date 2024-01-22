import sys
import os.path
from typing import List
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.chat_engine.types import ChatMode

PERSIST_DIR = "./storage"

def setup_chat_engine(docs_dir: str, exclude_list: List[str]):
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(docs_dir, recursive=True, exclude=exclude_list).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    chat_engine = index.as_retriever(chat_mode=ChatMode.CONTEXT, verbose=True) # type: ignore
    return chat_engine

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Supply prompt as cli arg.")
        exit(1)

    prompt = sys.argv[1]
    chat_engine = setup_chat_engine("../notes/", ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.svg"])
    response = chat_engine.retrieve(prompt)

    for i in response:
        print(i, end="\n\n")

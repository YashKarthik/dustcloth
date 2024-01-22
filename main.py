import sys
import os.path
import pickle
from typing import List
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.chat_engine.types import ChatMode
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SemanticSplitterNodeParser

PERSIST_DIR = "./storage"

def setup_engine(docs_dir: str, exclude_list: List[str]):
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(docs_dir, recursive=True, exclude=exclude_list).load_data()

        embed_model = OpenAIEmbedding()
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshhold=95,
            embed_model=embed_model
        )
        print("Splitting...")
        nodes = semantic_splitter.build_semantic_nodes_from_documents(documents, show_progress=True)
        non_empty_nodes = list(filter(lambda node: node, nodes))

        with open("./nodes.ob", "wb") as f:
            pickle.dump(non_empty_nodes, f)

        print("Creating index...")
        index = VectorStoreIndex(non_empty_nodes, show_progress=True)
        print("Saving index to disk...")
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
    chat_engine = setup_engine("../notes/", ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.svg"])
    response = chat_engine.retrieve(prompt)

    for i in response:
        print(i, end="\n\n")

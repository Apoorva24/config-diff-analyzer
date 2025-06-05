import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"
project_root = Path(__file__).parent
DATA_PATH = f"{project_root}/data"
VENDOR_DOCS_PATH= f"{DATA_PATH}/vendor_docs"
VENDORS = ["Arista", "Juniper"]

def build_vectorstore_for_vendor(vendor_name: str, input_folder: str, persist_dir: str):
    print(f"Processing vendor: {vendor_name}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_docs = []
    for file_path in Path(input_folder).rglob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        all_docs.extend(splitter.split_documents(docs))

    print(f"Total documents after split: {len(all_docs)}")
    Chroma.from_documents(documents=all_docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=persist_dir)
    print(f"Vectorstore created for {vendor_name} at {persist_dir}")

if __name__ == "__main__":
    for vendor in VENDORS:
        folder = f"{VENDOR_DOCS_PATH}/{vendor.lower()}"
        persist_path = f"{DATA_PATH}/chroma_{vendor.lower()}"
        build_vectorstore_for_vendor(vendor, folder, persist_path)
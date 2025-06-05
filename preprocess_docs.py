import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
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
    Chroma.from_documents(documents=all_docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key), persist_directory=persist_dir)
    print(f"Vectorstore created for {vendor_name} at {persist_dir}")

if __name__ == "__main__":
    for vendor in VENDORS:
        folder = f"{VENDOR_DOCS_PATH}/{vendor.lower()}"
        persist_path = f"{DATA_PATH}/chroma_{vendor.lower()}"
        build_vectorstore_for_vendor(vendor, folder, persist_path)
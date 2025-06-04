from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

DOCS_PATH = "/Users/apoorvah/config-diff-analyzer/data"
VENDOR_DOCS_PATH= "/Users/apoorvah/config-diff-analyzer/data/vendor_docs"
VENDORS = ["Arista", "Juniper"]

def build_vectorstore_for_vendor(vendor_name: str, input_folder: str, persist_dir: str):
    print(f"Processing vendor: {vendor_name}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_docs = []
    for file_path in Path(input_folder).rglob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        all_docs.extend(splitter.split_documents(docs))

    print(f"Total documents after split: {len(all_docs)}")
    Chroma.from_documents(documents=all_docs, embedding=embedding_model, persist_directory=persist_dir)
    print(f"Vectorstore created for {vendor_name} at {persist_dir}")

if __name__ == "__main__":
    for vendor in VENDORS:
        folder = f"{VENDOR_DOCS_PATH}/{vendor.lower()}"
        persist_path = f"{DOCS_PATH}/chroma_{vendor.lower()}"
        build_vectorstore_for_vendor(vendor, folder, persist_path)
import difflib
import os
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "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"
project_root = Path(__file__).parent
DATA_PATH = f"{project_root}/data"

# Load pre-built vectorstore
@st.cache_resource
def load_vectorstore(vendor):
    persist_path = f"./vectorstores/chroma_{vendor}"
    return Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

# UI
st.title("Config Diff Analyzer")

vendor = st.selectbox("Select device vendor", ["Arista", "Juniper"]) # Also add support for Cisco, Fortinet and other vendors
retriever = load_vectorstore(vendor).as_retriever(search_kwargs={"k": 3})

# Prompt and LLM setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior network engineer helping review config changes for routers and switches."),
    ("user", "Here is the diff between two configs:\n\n{diff}\n\nBased on the documentation:\n{context}\n\n{question}")
])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
chain = prompt | llm

device_config = st.file_uploader("Upload Device Config", type=["txt", "conf"], key="device_config")
desired_config = st.file_uploader("Upload Desired Config", type=["txt", "conf"], key="desired_config")
question = st.text_area("Enter your question about the diff", height=100, placeholder="e.g. What are the risks of applying this change without draining?")

if st.button("Analyze Diff"):
    if not device_config or not desired_config or not question.strip():
        st.warning("Please upload device and desired configs and ask a question about the diff.")
    else:
        device_config = device_config.read().decode("utf-8")
        desired_config = desired_config.read().decode("utf-8")
        diff = "\n".join(difflib.unified_diff(
            device_config.splitlines(), desired_config.splitlines(),
            fromfile="Device Config", tofile="Desired Config", lineterm=""
        ))
        st.subheader("Config Diff")
        st.code(diff, language="diff")

        with st.spinner("Retrieving context and analyzing..."):
            retrieved = retriever.get_relevant_documents(diff)
            context = "\n---\n".join([doc.page_content for doc in retrieved])
            response = chain.invoke({"diff": diff, "context": context, "question": question})

        st.subheader("Analysis Result")
        st.markdown(response.content)

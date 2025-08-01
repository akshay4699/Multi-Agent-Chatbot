import streamlit as st
import os

from huggingface_hub import login

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from typing import List, Literal
from langgraph.graph import StateGraph, END, START

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing_extensions import TypedDict

# --- UI Setup ---
st.set_page_config(page_title="LangGraph RAG Router", layout="wide")
st.title("🧠 LangGraph - RAG Routing with Groq + HuggingFace")

with st.sidebar:
    st.header("🔐 API Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("HuggingFace API Token", type="password")

    if st.button("🚀 Initialize App"):
        login(token=hf_token)
        st.session_state.initialized = True
        st.success("✅ Initialized successfully!")

# --- Initialize once ---
if "initialized" not in st.session_state:
    st.info("Please enter API keys in sidebar and click 'Initialize App'")
    st.stop()

# --- Load Docs ---
@st.cache_resource(show_spinner="Loading documents and creating vectorstore...")
def load_docs_and_store():
    urls = [
        'https://www.excel-easy.com/',
        'https://www.rib-software.com/en/blogs/bi-dashboard-design-principles-best-practices',
        'https://www.webdatarocks.com/blog/best-charts-discrete-data/'
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list = [doc for sublist in docs for doc in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(doc_list)[:50]

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5')
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore, texts

# Load vectorstore
vectorstore, loaded_texts = load_docs_and_store()
retriever = vectorstore.as_retriever()

# --- LLM for Routing and Answering ---
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)

class RouteQuery(BaseModel):
    datasource: Literal['vectorstore', 'wiki_search'] = Field(...)

system = """
You are an expert at routing a user question to a vectorstore or Wikipedia.
The vectorstore contains documents related to dashboards, data visualization techniques, and Excel formula guides.
Use the vectorstore for those topics. Use Wikipedia for everything else.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])
structured_route_query = llm.with_structured_output(RouteQuery)
question_router = route_prompt | structured_route_query

# --- Wikipedia Tool ---
api_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wiki)

# --- LangGraph Types ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def route_question(state):
    question = state['question']
    source = question_router.invoke({"question": question})
    return "vectorstore" if source.datasource == "vectorstore" else "wiki_search"

def retrieve(state):
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search_node(state):
    question = state['question']
    wiki_result = wiki_tool.run(question)
    document = Document(page_content=wiki_result, metadata={})
    return {"documents": [document], "question": question}

def generate_answer(state):
    question = state['question']
    docs = state['documents']
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Question: {question}\n\nContext:\n{context}"
    answer = llm.invoke(prompt).content
    return {"question": question, "documents": docs, "generation": answer}

# --- Build LangGraph ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("wiki_search", wiki_search_node)
workflow.add_node("generate", generate_answer)

workflow.add_conditional_edges(
    START, route_question,
    {"vectorstore": "retrieve", "wiki_search": "wiki_search"}
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("wiki_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- Run Query ---
st.divider()
st.subheader("🔍 Ask your Question")

user_question = st.text_input("Type your question:")
if st.button("Submit") and user_question:
    with st.spinner("Thinking..."):
        result = app.invoke({"question": user_question})
        st.markdown("### 🤖 Answer")
        st.write(result["generation"])
        st.markdown("---")
        st.markdown("### 📄 Source Documents")
        for doc in result["documents"]:
            st.write(doc.page_content[:500] + "...")

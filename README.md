# 🧠 Multi-Agent Chatbot Assistant

An intelligent chatbot system using a **multi-agent LangGraph architecture**, combining:
- 🔍 Custom FAISS Vector Database
- 🌐 Live Wikipedia Search Tool
- 🧩 Dynamic Tool Routing with LangChain
- 🖼️ Streamlit Frontend for Instant Access

🔗 [Live App Demo](https://multi-agent-chatbot-afvxqxttyindcaqgaku9kc.streamlit.app/)

## 🔧 Technologies Used
- LangGraph Agents (LangChain)
- FAISS Vector Store
- Wikipedia Tool via LangChain Tools
- HuggingFace Embedding Technique
- GROQ LLM
- Streamlit (for UI)

## 🧩 Features
- 📚 Private Knowledge QA via FAISS
- 🌍 Real-time Info from Wikipedia
- 🗂️ Intelligent Routing via `Multi-Agent LangGraph`
- 🖥️ Minimalist Web UI

## 🚀 Setup Instructions

```bash
git clone https://github.com/yourusername/multi-agent-chatbot.git
cd multi-agent-chatbot
pip install -r requirements.txt

# Add your environment variables
cp .env.example .env
# Add your keys (OpenAI/Groq, etc.)

streamlit run app.py

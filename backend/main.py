import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

from retriever import build_retriever, retrieve_with_parent
from tools import agri_tools

load_dotenv()

app = FastAPI(title="AgriChatbot API 🌾")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

# ── Retriever ──
print("⏳ Loading retriever...")
retriever, vectorstore = build_retriever(llm)
print("✅ Retriever ready!")

# ── RAG tool ──
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the agricultural knowledge base (PDFs) for information about
    crops, soil, fertilizers, irrigation, pest management, farming techniques.
    Always try this FIRST before web search.
    """
    docs = retriever.invoke(query)
    docs = retrieve_with_parent(docs)
    if not docs:
        return "No relevant information found in the knowledge base."
    result = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown").split("\\")[-1]
        page   = doc.metadata.get("page", "?")
        result += f"[Source {i}: {source}, page {page}]\n{doc.page_content}\n\n"
    return result

all_tools = [search_knowledge_base] + agri_tools

# ── System prompt ──
system_prompt = """You are AgriBot 🌾, an expert agricultural assistant for Indian farmers.

You help with:
- Crop selection, sowing, and harvesting advice
- Soil health, fertilizers, and irrigation
- Pest and disease identification and management
- Weather-based farming decisions
- Mandi prices and market information
- Government schemes and subsidies

RULES:
1. ALWAYS use search_knowledge_base tool first.
2. If not found in knowledge base, use web_search tool.
3. For weather queries, use get_weather tool.
4. For price queries, use get_mandi_price tool.
5. Always mention source PDF when answering from knowledge base.
6. If question is NOT agriculture related, politely decline.
7. Always give DETAILED and COMPLETE answers. Never cut short.
8. For prices, mention multiple markets, min/max/modal prices.
9. Respond in the same language as the user (Hindi or English).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Agent ──
agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_execution_time=60,
)

# ── Models ──
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []

# ── Routes ──
@app.get("/")
def root():
    return {"status": "AgriChatbot API is running 🌾"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history = []
        for msg in request.chat_history[-6:]:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))

        result = agent_executor.invoke({
            "input"        : request.message,
            "chat_history" : history,
        })

        answer = result.get("output", "Sorry, I could not process your request.")

        # Extract sources from intermediate steps
        seen_urls = set()
        seen_domains = set()
        all_urls = []
        intermediate_steps = result.get("intermediate_steps", [])
        for action, observation in intermediate_steps:
            tool_name = action.tool
            if tool_name in ["web_search", "get_mandi_price"]:
                lines = str(observation).split("\n")
                for line in lines:
                    if "🔗 Source:" in line:
                        url = line.replace("🔗 Source:", "").strip()
                        try:
                            domain = url.split("/")[2].replace("www.", "")
                        except:
                            domain = url
                        # Deduplicate by both full URL and domain
                        if url not in seen_urls and domain not in seen_domains:
                            seen_urls.add(url)
                            seen_domains.add(domain)
                            all_urls.append(url)

        # Build clean sources section
        sources_text = ""
        if all_urls:
            sources_text = "\n\n---\n📚 **Sources:**\n"
            for i, url in enumerate(all_urls, 1):
                try:
                    domain = url.split("/")[2].replace("www.", "")
                except:
                    domain = url
                sources_text += f"{i}. 🔗 [{domain}]({url})\n"

        final_answer = answer + sources_text
        return ChatResponse(answer=final_answer)

    except Exception as e:
        return ChatResponse(answer=f"Sorry, an error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
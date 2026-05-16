import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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


    # ── LLM with 8-model fallback ──
def get_llm():
    models = [
        # Priority 1 — Groq LLaMA 3.3 70b (best quality + fastest)
        ("Groq LLaMA-3.3-70b", lambda: ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )),
        # Priority 2 — Gemini 2.0 Flash (cross-provider, excellent quality)
        ("Gemini 2.0 Flash", lambda: ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )),
        # Priority 3 — Groq LLaMA3 70b (separate quota from #1)
        ("Groq LLaMA3-70b", lambda: ChatGroq(
            model="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )),
        # Priority 4 — Groq Gemma2 9b (500K tokens/day quota)
        ("Groq Gemma2-9b", lambda: ChatGroq(
            model="gemma2-9b-it",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )),
        # Priority 5 — Groq Mixtral (500K tokens/day quota)
        ("Groq Mixtral", lambda: ChatGroq(
            model="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )),
        # Priority 6 — Gemini 1.5 Flash (extra Google buffer)
        ("Gemini 1.5 Flash", lambda: ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )),
        # Priority 7 — HF Qwen 2.5 72B (unlimited)
        ("HF Qwen2.5-72B", lambda: ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-72B-Instruct",
                huggingfacehub_api_token=os.getenv("HF_API_KEY"),
                temperature=0.1,
                max_new_tokens=1024,
            )
        )),
        # Priority 8 — HF LLaMA 3.3 70B (unlimited, last resort)
        ("HF LLaMA-3.3-70B", lambda: ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                huggingfacehub_api_token=os.getenv("HF_API_KEY"),
                temperature=0.1,
                max_new_tokens=1024,
            )
        )),
    ]

    for name, model_fn in models:
        try:
            llm = model_fn()
            print(f"✅ LLM ready: {name}")
            return llm, name
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")
            continue

    raise RuntimeError("❌ All LLMs failed!")

llm, active_model = get_llm()







# ── Retriever ──
retriever = None
vectorstore = None
try:
    print("⏳ Loading retriever...")
    retriever, vectorstore = build_retriever(llm)
    print("✅ Retriever ready!")
except Exception as e:
    print(f"❌ Retriever failed: {e}")
    import traceback
    traceback.print_exc()

# ── RAG tool ──
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the agricultural knowledge base (PDFs) for information about
    crops, soil, fertilizers, irrigation, pest management, farming techniques.
    Always try this FIRST before web search.
    """
    if retriever is None:
        return "Knowledge base unavailable."
    try:
        docs = retriever.invoke(query)
        docs = retrieve_with_parent(docs)
        if not docs:
            return "No relevant information found in the knowledge base."
        result = ""
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
            page   = doc.metadata.get("page", "?")
            result += f"[Source {i}: {source}, page {page}]\n{doc.page_content}\n\n"
        return result
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

all_tools = [search_knowledge_base] + agri_tools

# ── System prompt ──
system_prompt = """You are AgriBot, an expert agricultural assistant for Indian farmers.

You have access to these tools:
- search_knowledge_base: Search PDF knowledge base FIRST for any farming question
- get_weather: Get weather for a location
- get_mandi_price: Get crop market prices
- web_search: Search web if knowledge base has no answer

STRICT RULES:
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
    global llm, active_model, agent, agent_executor
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

        # Extract sources
        seen_urls    = set()
        seen_domains = set()
        all_urls     = []
        for action, observation in result.get("intermediate_steps", []):
            if action.tool in ["web_search", "get_mandi_price"]:
                for line in str(observation).split("\n"):
                    if "🔗 Source:" in line:
                        url = line.replace("🔗 Source:", "").strip()
                        try:
                            domain = url.split("/")[2].replace("www.", "")
                        except:
                            domain = url
                        if url not in seen_urls and domain not in seen_domains:
                            seen_urls.add(url)
                            seen_domains.add(domain)
                            all_urls.append(url)

        sources_text = ""
        if all_urls:
            sources_text = "\n\n---\n📚 **Sources:**\n"
            for i, url in enumerate(all_urls, 1):
                try:
                    domain = url.split("/")[2].replace("www.", "")
                except:
                    domain = url
                sources_text += f"{i}. 🔗 [{domain}]({url})\n"

        return ChatResponse(answer=answer + sources_text)

    except Exception as e:
        error_msg = str(e).lower()

        # Rate limit hit — switch to next model automatically
        if "rate_limit" in error_msg or "429" in error_msg or "quota" in error_msg:
            print(f"⚠️ Rate limit on {active_model} — switching model...")
            try:
                llm, active_model = get_llm()
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
                # Retry with new model
                result = agent_executor.invoke({
                    "input"        : request.message,
                    "chat_history" : [],
                })
                answer = result.get("output", "Sorry, try again.")
                return ChatResponse(answer=f"[Switched to {active_model}]\n\n" + answer)
            except Exception as e2:
                return ChatResponse(answer="All AI models are currently busy. Please try again in a few minutes. 🙏")

        return ChatResponse(answer=f"Sorry, an error occurred: {str(e)}")
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

        # Extract sources
        seen_urls    = set()
        seen_domains = set()
        all_urls     = []
        for action, observation in result.get("intermediate_steps", []):
            if action.tool in ["web_search", "get_mandi_price"]:
                for line in str(observation).split("\n"):
                    if "🔗 Source:" in line:
                        url = line.replace("🔗 Source:", "").strip()
                        try:
                            domain = url.split("/")[2].replace("www.", "")
                        except:
                            domain = url
                        if url not in seen_urls and domain not in seen_domains:
                            seen_urls.add(url)
                            seen_domains.add(domain)
                            all_urls.append(url)

        sources_text = ""
        if all_urls:
            sources_text = "\n\n---\n📚 **Sources:**\n"
            for i, url in enumerate(all_urls, 1):
                try:
                    domain = url.split("/")[2].replace("www.", "")
                except:
                    domain = url
                sources_text += f"{i}. 🔗 [{domain}]({url})\n"

        return ChatResponse(answer=answer + sources_text)

    except Exception as e:
        return ChatResponse(answer=f"Sorry, an error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
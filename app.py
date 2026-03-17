import os, json
import requests
from urllib.parse import urljoin, unquote
import re
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ========= إعدادات =========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
# Notice: Open source model from sentence-transformers
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

API_BASE    = os.getenv("API_BASE", "").rstrip("/")
API_AUTH    = os.getenv("API_AUTH", "")

API_HEADERS: Dict[str, str] = {}
_headers_env = os.getenv("API_HEADERS")
if _headers_env:
    try:
        API_HEADERS = dict(json.loads(_headers_env))
    except Exception:
        pass

if API_AUTH:
    API_HEADERS["Authorization"] = API_AUTH

# ========= LangChain LLM & Embeddings =========
# Open Source Chat via Groq
if GROQ_API_KEY:
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL, temperature=0.3)
else:
    llm = None

# Open Source Embeddings via HuggingFace (Runs Locally)
# We avoid instantiating it globally to prevent large downloads on startup if not needed 
# But for FAISS we typically instantiate it once
embeddings_client = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ========= بحث المتجهات (FAISS) =========
import faiss
import numpy as np

INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")
DOCS_PATH  = os.getenv("DOCS_PATH",  "data/chunks.jsonl")

# تحميل الفهرس والمستندات إن وجدوا
index = None
chunks: List[Dict[str, Any]] = []
if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    # WARNING: If this index was created using OpenAI/Gemini embeddings, 
    # it WILL NOT work with the new HuggingFace embedding sizes.
    # We load it anyway but it will likely produce incorrect scores/crashing if dimensions mismatch.
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        index = None
        chunks = []

# ========= موديلات الطلب/الاستجابة =========
class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 4
    temperature: float = 0.3

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

app = FastAPI(title="AI Chatbot (LangChain Open Source)")

# ========= CORS =========
origins = os.getenv("CORS_ALLOW", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= أدوات مساعدة =========
def embed_query(query: str) -> np.ndarray:
    vec = embeddings_client.embed_query(query)
    v = np.array(vec, dtype="float32")
    norm = np.linalg.norm(v)
    if norm != 0: 
        v = v / norm
    return np.array([v])

def retrieve_from_faiss(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    if index is None or not chunks:
        return []
        
    try:
        qv = embed_query(query)
        D, I = index.search(qv, top_k) # type: ignore
        results = []
        
        distances = getattr(D, "__getitem__", lambda x: [])(0) if D is not None and getattr(D, "__len__", lambda: 0)() > 0 else []
        indices = getattr(I, "__getitem__", lambda x: [])(0) if I is not None and getattr(I, "__len__", lambda: 0)() > 0 else []
        
        for dist, idx in zip(distances, indices):
            if idx == -1: 
                continue
            idx_int = int(idx)
            if 0 <= idx_int < len(chunks):
                ch = chunks[idx_int]
                results.append({
                    "score": float(dist),
                    "text": ch.get("text", ""),
                    "source": ch.get("source", ""),
                    "title": ch.get("title", "")
                })
        return results
    except Exception as e:
        print(f"Error during FAISS retrieval (Possible dimension mismatch due to Model change): {e}")
        return []

# ========= API Helpers =========
def live_api_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not API_BASE: 
        return []
    out = []
    for endpoint in ["posts", "pages", "categories"]:
        url = urljoin(API_BASE + "/", endpoint + "?search=" + requests.utils.quote(query))
        try:
            r = requests.get(url, headers=API_HEADERS, params={"per_page": limit})
            if r.status_code == 200:
                arr = r.json()
                if not isinstance(arr, list):
                    continue
                if endpoint == "categories":
                    for cat in arr:
                        if not isinstance(cat, dict): continue
                        cat_name = cat.get("name", "")
                        cat_desc = cat.get("description", "")
                        cat_link = cat.get("link", "")
                        if query.lower() in cat_name.lower() or query.lower() in cat_desc.lower():
                            cat_text = f"التصنيف: {cat_name}\n{cat_desc}".strip()
                            if cat_text:
                                out.append({"text": cat_text, "meta": {"title": f"تصنيف: {cat_name}", "url": cat_link}})
                else:
                    for it in arr:
                        if not isinstance(it, dict): continue
                        title = it.get("title", {}).get("rendered", "")
                        link = it.get("link", "")
                        content_html = it.get("content", {}).get("rendered", "")
                        excerpt = it.get("excerpt", {}).get("rendered", "")
                        
                        content_clean = re.sub(r'<[^>]+>', '', content_html)
                        content_clean = re.sub(r'\s+', ' ', content_clean).strip()
                        
                        if len(content_clean) < 100 and excerpt:
                            excerpt_clean = re.sub(r'<[^>]+>', '', excerpt)
                            content_clean = re.sub(r'\s+', ' ', excerpt_clean).strip()
                            
                        if not content_clean or len(content_clean) < 20: 
                            continue
                            
                        text = f"{title}\n\n{content_clean}".strip()
                        out.append({"text": text, "meta": {"title": title, "url": link}})
        except Exception as e:
            print(f"Error fetching from API: {e}")
            pass
    return out

# ========= قالب الاستجابة بـ LangChain =========
system_template = """
أنت Chatbot ذكي ومساعد لشركة إعلانك، متخصص في التسويق الإلكتروني والذكاء الاصطناعي.
اعتمد دائمًا على المصادر المقتبسة من موقع الشركة والمعرفة العامة في التسويق.
إذا كانت المعلومات غير متوفرة في المصادر، استخدم خبرتك في التسويق الإلكتروني لتقديم إجابة مفيدة واستخدم اللغة العربية الفصحى المبسطة.
تأكد من ذكر الروابط المصدرية دائماً بناء على المصادر المتاحة أسفله. لا تقل 'لا أملك' أو 'لا أعرف'.

المصادر المتاحة:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}")
])

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not llm:
         return ChatResponse(answer="عذراً، لم يتم إعداد مفتاح API الخاص بـ Groq. الرجاء إضافته للـ .env", sources=[])
    
    ctx_faiss = retrieve_from_faiss(req.message, top_k=4)
    ctx_live = live_api_search(req.message, limit=3)
    
    formatted_contexts = []
    light_sources: List[Dict[str, Any]] = []
    
    # تنسيق نتائج الـ API Live
    for res in ctx_live:
        title = res.get("meta", {}).get("title", "")
        url = res.get("meta", {}).get("url", "")
        text_content = res.get("text", "")
        formatted_contexts.append(f"[المصدر {len(formatted_contexts)+1}] {title}\nالرابط: {url}\nالمحتوى: {text_content[:800]}...")
        light_sources.append({"title": title, "url": url, "score": 1.0})
        
    # تنسيق نتائج الـ FAISS
    for res in ctx_faiss:
        title = res.get("title", "بدون عنوان")
        url = res.get("source", "")
        text_content = res.get("text", "")
        formatted_contexts.append(f"[المصدر {len(formatted_contexts)+1}] {title}\nالرابط: {url}\nالمحتوى: {text_content[:800]}...")
        light_sources.append({"title": title, "url": url, "score": res.get("score", 0.0)})
        
    context_str = "\n\n".join(formatted_contexts) if formatted_contexts else "لم يتم العثور على مصادر محددة من الموقع."
    
    # LangChain Chat Sequence
    # نمرر درجة الحرارة المحددة من المستخدم كمتغير
    llm_with_temp = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL, temperature=req.temperature)
    chain = prompt_template | llm_with_temp
    
    try:
        response = chain.invoke({
            "context": context_str,
            "question": req.message
        })

        answer = response.content if hasattr(response, "content") else str(response)
        if isinstance(answer, str):
            answer = answer.strip()
            answer = unquote(answer)
        else:
            answer = str(answer) # fallback
    except Exception as e:
        answer = f"عفواً، حدث خطأ أثناء الاتصال بالخادم المفتوح المصدر (Groq). الخطأ: {str(e)}"
    
    sources_to_return = [s for i, s in enumerate(light_sources) if i < 7]
    return ChatResponse(answer=answer, sources=sources_to_return)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/test-api")
def test_api():
    """اختبار الـ API واستخراج الـ links"""
    try:
        test_query = "التسويق الإلكتروني"
        results = live_api_search(test_query, limit=3)
        
        return {
            "query": test_query,
            "results_count": len(results),
            "results": [
                {
                    "title": res.get("meta", {}).get("title", ""),
                    "url": res.get("meta", {}).get("url", ""),
                    "type": res.get("meta", {}).get("type", "post"),
                    "text_preview": res.get("text", "")[:200] + "..."
                }
                for res in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

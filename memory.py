# core/memory.py
import json
import os
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from core.mind import llm

# Pfade & Parameter
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "longterm_memory.json")
EPISODIC_FILE = os.path.join(os.path.dirname(__file__), "episodic_memory.json")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")
SUMMARY_LIMIT = 20

# Embedding-Modell für Vektorstore
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Vektorstore initialisieren
if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    # Leerer Vektorstore bei Erststart
    vectorstore = FAISS.from_documents([], embedding)
    vectorstore.save_local(VECTORSTORE_PATH)

# Retriever erzeugen
retriever = vectorstore.as_retriever()

# Langzeitgedächtnis laden/speichern
def load_memory() -> List[BaseMessage]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    for item in data:
        if item["type"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            messages.append(AIMessage(content=item["content"]))
        elif item["type"] == "summary":
            messages.append(SystemMessage(content=f"[Summary of older conversations]\n{item['content']}"))
    return messages


def save_memory(messages: List[BaseMessage]):
    data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            data.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"type": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            data.append({"type": "summary", "content": msg.content})

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Zusammenfassen für Langzeitgedächtnis

def summarize_and_trim(messages: List[BaseMessage]) -> List[BaseMessage]:
    if len(messages) <= SUMMARY_LIMIT:
        return messages

    base_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    to_summarize = base_msgs[:-SUMMARY_LIMIT]
    keep = base_msgs[-SUMMARY_LIMIT:]

    if not to_summarize:
        return messages

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation. Stick to the essentials, but retain personal details."),
        ("human", "{text}")
    ])
    summary_chain: Runnable = summary_prompt | llm

    full_text = "\n".join([m.content for m in to_summarize])
    summary = summary_chain.invoke({"text": full_text})

    summarized_messages = [SystemMessage(content=summary.content)] + keep
    return summarized_messages


def summarize_messages(messages: List[BaseMessage]) -> str:
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation. Stick to the essentials, but retain personal details."),
        ("human", "{text}")
    ])
    summary_chain = summary_prompt | llm
    text = "\n".join([m.content for m in messages if isinstance(m, (HumanMessage, AIMessage))])
    result = summary_chain.invoke({"text": text})
    return result


# Episodenspeicherung

def load_episodes() -> List[dict]:
    if not os.path.exists(EPISODIC_FILE):
        return []
    with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episode(title: str, summary: str, messages: List[BaseMessage]):
    episodes = load_episodes()
    episodes.append({
        "title": title,
        "summary": summary,
    })
    with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    
    add_to_vectorstore(title, summary)

# Vektorstore: Hinzufügen & Suchen

def add_to_vectorstore(title: str, content: str):
    doc = Document(page_content=content, metadata={"title": title})
    vectorstore.add_documents([doc])
    vectorstore.save_local(VECTORSTORE_PATH)


def search_vectorstore(query: str, k: int = 3) -> List[str]:
    results = retriever.invoke(query)
    return [doc.page_content for doc in results[:k]]


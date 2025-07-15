import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.graph import StateGraph, END
from mem0 import MemoryClient
from typing import TypedDict, List, Any
from datetime import datetime, timedelta, timezone
import requests
import tempfile
import os

os.environ["MEM0_API_KEY"] = "m0-bFj5UTXVlpNpSnlaDOfyrfTdYiAWQxcYiQbOHqxk"

client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

# ✅ Criação segura da memória (caso não exista)
try:
    client.create(memory_id="agente_estudos", user_id="usuario1", description="Memória do Agente de Estudos")
    print("🧠 Memória 'agente_estudos' criada com sucesso.")
except Exception as e:
    print("ℹ️ Memória já existe ou erro ignorável:", e)

def call_llm(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    st.write("Resposta bruta da API:", res.text)
    return res.json().get("response", "[Erro ao gerar resposta]")

def split_document(document, embedding_model):
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    return splitter.split_documents(document)

def search_relevant_memories(query):
    results = client.search(query=query, user_id="usuario1", memory_id="agente_estudos")
    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    recent = [r for r in results if datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc) >= last_week]
    return "\n".join([r.get("message", r.get("content", "")) for r in recent])

def classification(question):
    classifier = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "Você é um classificador de mensagens. Após analisar a mensagem, responda apenas com: explicar, gerar_pergunta ou avaliar."},
                {"role": "user", "content": question}
            ]
        }
    )

    try:
        full_text = classifier.text.strip().lower()
        print("DEBUG - Texto bruto do classificador:", full_text)
        if "explicar" in full_text:
            return "explicar"
        elif "gerar_pergunta" in full_text:
            return "gerar_pergunta"
        elif "avaliar" in full_text:
            return "avaliar"
        else:
            st.write("⚠️ Classificação inesperada, usando fallback.")
            return "explicar"
    except Exception as e:
        st.write("Erro no classificador:", e)
        return "explicar"

def route_input(state):
    question = state["pergunta"]
    rota = classification(question).strip()
    st.write(f"DEBUG - Rota classificada: '{rota}'")
    return rota

def explanation_node(state):
    question = state["pergunta"]
    docs = state["retriever"].get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    history = search_relevant_memories(question)
    prompt = f"Contexto:\n{context}\n\nHistórico:\n{history}\n\nPergunta:\n{question}"
    st.write("PROMPT FINAL:", prompt)
    answer = call_llm(prompt)
    st.write("RESPOSTA LLM:", answer)
    client.add(
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        user_id="usuario1",
        memory_id="agente_estudos"
    )
    return {"resposta": answer}

def question_generation_node(state):
    docs = state["retriever"].get_relevant_documents("Conteúdo Importante!")
    context = "\n".join([doc.page_content for doc in docs])
    history = search_relevant_memories("gerar perguntas")
    prompt = f"Gere uma pergunta de múltipla escolha sobre o conteúdo:\n{context}\n\nHistórico:\n{history}"
    question = call_llm(prompt)
    client.add(
        messages=[{"role": "user", "content": "Me pergunte algo"}, {"role": "assistant", "content": question}],
        user_id="usuario1",
        memory_id="agente_estudos"
    )
    st.write("DEBUG - Resposta LLM:", question)
    return {"resposta": question}

def answere_evaluator_node(state):
    question = state["pergunta"]
    user_answere = state.get("resposta_usuario", "")
    docs = state["retriever"].get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Contexto:\n{context}\n\nPergunta:\n{question}\nResposta do usuário:\n{user_answere}\nAvalie se a resposta está correta e justifique."
    evaluation = call_llm(prompt)
    client.add(
        messages=[{"role": "user", "content": f"{question}\nResposta: {user_answere}"}, {"role": "assistant", "content": evaluation}],
        user_id="usuario1",
        memory_id="agente_estudos"
    )
    return {"avaliacao": evaluation}

class GraphState(TypedDict):
    pergunta: str
    retriever: Any
    resposta_usuario: str
    memoria: List[Any]

st.title("Agente de Estudos!")
upload_file = st.file_uploader("Envie um PDF", type=".pdf")

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(upload_file.read())
        pdf_path = tmp.name

    with st.spinner("Carregando e processando documento..."):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
        chunks = split_document(documents, embedding_model)
        db_faiss = FAISS.from_documents(chunks, embedding_model)
        retriever = db_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    user_input = st.text_input("Digite sua pergunta ou peça por uma questão: ")
    user_answere = st.text_input("(Opcional) Escreva sua resposta para ser avaliada")

    if st.button("Executar") and user_input:
        graph = StateGraph(GraphState)
        graph.add_node("explicar", explanation_node)
        graph.add_node("gerar_pergunta", question_generation_node)
        graph.add_node("avaliar", answere_evaluator_node)
        graph.set_conditional_entry_point(route_input)
        graph.add_edge("explicar", END)
        graph.add_edge("gerar_pergunta", END)
        graph.add_edge("avaliar", END)
        flow = graph.compile()

        initial_state = {
            "pergunta": user_input,
            "retriever": retriever,
            "resposta_usuario": user_answere,
            "memoria": []
        }

        resultado = flow.invoke(initial_state)

        st.subheader("DEBUG: Resultado bruto")
        st.write(resultado)

        # if "resposta" in resultado:
        #     st.subheader("Resposta")
        #     st.write(resultado["resposta"])
        # if "avaliacao" in resultado:
        #     st.subheader("Avaliação da Resposta")
        #     st.write(resultado["avaliacao"])

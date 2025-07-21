"""
Agente de Estudos com LangChain, LangGraph, HuggingFace e mem0
Organizado com princípios de Clean Code e SOLID.
"""

# ============================
# Imports
# ============================
import os
import json
import tempfile
from datetime import datetime, timedelta, timezone
from typing import TypedDict, List, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.graph import StateGraph, END
from mem0 import MemoryClient


# ============================
# Configurações Iniciais
# ============================
os.environ['MEM0_API_KEY'] = "m0-bFj5UTXVlpNpSnlaDOfyrfTdYiAWQxcYiQbOHqxk"
os.environ['HF_TOKEN'] = ""
load_dotenv('./env')
HF_TOKEN = os.getenv("HF_TOKEN")
client = MemoryClient(api_key=os.environ['MEM0_API_KEY'])

# ============================
# Classes
# ============================
class GraphState(TypedDict):
    pergunta: str
    retriever: Any
    resposta_usuario: str
    memoria: List[Any]
    resposta: str
    avaliacao: str


# ============================
# Funções utilitárias
# ============================
def inicializar_memoria():
    try:
        client.create(memory_id="agente_estudos", user_id="usuario1", description="Memória do Agente de Estudos")
    except Exception:
        pass

def call_llm(prompt: str):
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-3-12b-it",
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.03,
        huggingface_api_token=HF_TOKEN
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model.invoke(prompt)

def split_document(document, embedding_model):
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    return splitter.split_documents(document)

def search_relevant_memories(query: str):
    results = client.search(query=query, user_id="usuario1", memory_id="agente_estudos")
    last_week = datetime.now(timezone.utc) - timedelta(days=7)
    recent = [r for r in results if datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc) >= last_week]
    return "\n".join([r.get("message", r.get("memory", "")) for r in recent])

def salvar_memoria(mensagem_salva):
    client.add(messages=mensagem_salva, user_id="usuario1", memory_id="agente_estudos")

def list_all_memories():
    termos = ["a", "pergunta", "resposta", "explica", "conclusão"]
    vistos, resultados = set(), []
    for termo in termos:
        try:
            mems = client.search(query=termo, user_id="usuario1", memory_id="agente_estudos")
            for m in mems:
                if m.get("id") not in vistos:
                    resultados.append(m)
                    vistos.add(m.get("id"))
        except:
            continue
    return resultados


# ============================
# Funções de Roteamento
# ============================
def classification(question):
    prompt = f"Você é um classificador. Analise: '{question}' e responda apenas com: explicar, gerar_pergunta ou avaliar."
    result = call_llm(prompt)
    texto = getattr(result, "content", str(result)).lower().strip()
    for tipo in ["explicar", "gerar_pergunta", "avaliar"]:
        if tipo in texto:
            return tipo
    return "explicar"

def route_input(state):
    return classification(state["pergunta"])


# ============================
# Nós do Grafo
# ============================
def explanation_node(state):
    question = state["pergunta"]
    docs = state["retriever"].get_relevant_documents(question)
    contexto = "\n".join([doc.page_content for doc in docs])
    historico = search_relevant_memories(question)
    prompt = f"""Você é um assistente educacional inteligente e prestativo da plataforma Cicero Tech AI, criado para ajudar estudantes a entender conteúdos acadêmicos com clareza, profundidade e foco. Sua missão é gerar explicações completas e acessíveis, respeitando o nível do usuário e sempre utilizando o conteúdo presente no material de estudo.

---
📚 Instruções de comportamento:
- Seja claro, direto e didático.  
- Nunca invente informações que não estejam no contexto ou histórico.  
- Se não houver informações suficientes, indique isso educadamente e oriente o usuário a revisar o material.  
- Utilize exemplos sempre que possível.  
- Evite respostas genéricas ou vagas.  
- Não inclua frases como “sou um assistente” ou “sou uma IA”, apenas entregue o conteúdo como um tutor humano faria.  

---
🧠 Contexto do conteúdo extraído do PDF:  
{contexto}

📜 Histórico recente de interações relevantes:  
{historico}

❓ Pergunta feita pelo usuário:  
{question}

---
🔍 Agora, com base no conteúdo e no histórico, explique a resposta à pergunta acima de forma clara, precisa e amigável. Use exemplos se forem úteis."""


    resposta = getattr(call_llm(prompt), "content", str(call_llm(prompt)))
    salvar_memoria([{"role": "user", "content": question}, {"role": "assistant", "content": resposta}])
    return {"resposta": resposta}

def question_generation_node(state):
    docs = state["retriever"].get_relevant_documents("Conteúdo Importante!")
    contexto = "\n".join([doc.page_content for doc in docs])
    historico = search_relevant_memories("gerar perguntas")
    prompt = f"Gere uma pergunta de múltipla escolha:\n{contexto}\n\nHistórico:\n{historico}"
    questao = getattr(call_llm(prompt), "content", str(call_llm(prompt)))
    salvar_memoria([{"role": "user", "content": "Me pergunte algo"}, {"role": "assistant", "content": questao}])
    return {"resposta": questao}

def answere_evaluator_node(state):
    question = state["pergunta"]
    resposta_usuario = state.get("resposta_usuario", "")
    docs = state["retriever"].get_relevant_documents(question)
    contexto = "\n".join([doc.page_content for doc in docs])
    prompt = f"Contexto:\n{contexto}\n\nPergunta:\n{question}\nResposta:\n{resposta_usuario}\nAvalie se está correta e justifique."
    avaliacao = getattr(call_llm(prompt), "content", str(call_llm(prompt)))
    salvar_memoria([{"role": "user", "content": f"{question}\nResposta: {resposta_usuario}"}, {"role": "assistant", "content": avaliacao}])
    return {"avaliacao": avaliacao}


# ============================
# Interface Streamlit
# ============================
st.title("Agente de Estudos!")
inicializar_memoria()

upload_file = st.file_uploader("Envie um PDF", type=".pdf", key="pdf_uploader_unique")

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(upload_file.read())
        caminho_pdf = tmp.name

    with st.spinner("Processando o documento..."):
        loader = PyMuPDFLoader(caminho_pdf)
        docs = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
        chunks = split_document(docs, embeddings)
        retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_type="similarity", search_kwargs={"k": 4})

    pergunta = st.text_input("Digite sua pergunta ou solicite uma questão:")
    resposta_usuario = st.text_input("(Opcional) Sua resposta para avaliação:")

    if st.button("Executar") and pergunta:
        grafo = StateGraph(GraphState)
        grafo.add_node("explicar", explanation_node)
        grafo.add_node("gerar_pergunta", question_generation_node)
        grafo.add_node("avaliar", answere_evaluator_node)
        grafo.set_conditional_entry_point(route_input)
        grafo.add_edge("explicar", END)
        grafo.add_edge("gerar_pergunta", END)
        grafo.add_edge("avaliar", END)
        fluxo = grafo.compile()

        estado_inicial = {
            "pergunta": pergunta,
            "retriever": retriever,
            "resposta_usuario": resposta_usuario,
            "memoria": [],
            "resposta": "",
            "avaliacao": ""
        }

        resultado = fluxo.invoke(estado_inicial)

        if "resposta" in resultado:
            st.subheader("Resposta")
            st.write(resultado["resposta"])
        if "avaliacao" in resultado:
            st.subheader("Avaliação da Resposta")
            st.write(resultado["avaliacao"])

    if st.button("🔍 Ver Memória"):
        resultados = list_all_memories()
        if resultados:
            st.subheader(f"🧠 Memórias ({len(resultados)} itens):")
            for r in resultados:
                conteudo = r.get("message", r.get("memory", ""))
                st.markdown(f"- **{r['created_at']}** ➜ {conteudo[:200]}...")
        else:
            st.warning("Nenhuma memória encontrada.")

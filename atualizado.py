import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any
import requests
import tempfile
import os

def call_llm(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt":prompt,
            "stream":False
        }
    )
    return res.json()#["response"]

def split_document(document, embedding_model):
    splitter = SemanticChunker(    
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90  )

    return splitter.split_documents(document)

memory = ConversationBufferMemory(return_messages = True)

def explanation_node(state):
    question = state["pergunta"]
    docs = state["retriever"].get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    history = "\n".join([msg.content for msg in state["memoria"]])
    prompt = f"Contexto:\n{context}\n\nHistórico:\n{history}\n\nPergunta:\n{question}"
    answer = call_llm(prompt)
    state["memoria"].append({'type':'human', 'data':question})
    state["memoria"].append({'type':'ia', 'data':answer})
    return {"resposta":answer, "memoria":state["memoria"]}

def question_generation_node(state):
    docs = state["retriever"].get_relevant_documents("Conteúdo Importante!")
    context = "\n".join([doc.page_content for doc in docs])
    history = "\n".join([msg["data"] for msg in state["memoria"]])
    prompt = f"Gere uma pergunta de múltipla escolha sobre o conteúdo:\n{context}\n\nHistórico:\n{history}"
    question = call_llm(prompt)
    state["memoria"].append({'type':'ia', 'data':question})
    return {"resposta":question, "memoria": state["memoria"]}

def answere_evaluator_node(state):
    question = state["pergunta"]
    user_answere = state.get("resposta_usuario", "")
    context = "\n".join([doc.page_content for doc in state["retriever"].get_relevant_documents(question)])
    prompt = f"Contexto:\n{context}\n\nPergunta:\n{question}\nResposta do usuário:\n{user_answere}\nAvalie se a resposta está correta e justifique."
    evaluation = call_llm(prompt)
    return {"avaliacao":evaluation, "memoria":state["memoria"]}


def route_input(state):
    question = state["pergunta"].lower()
    if "pergunta" in question or "questão" in question or "me pergunte" in question:
        return "gerar pergunta"
    elif "avaliar" in question or "corrigir" in question:
        return "avaliar"
    return "explicar"

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

        embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",
                                            model_kwargs={"trust_remote_code": True})
        chunks = split_document(documents, embedding_model)
        db_faiss = FAISS.from_documents(chunks, embedding_model)

        retriever = db_faiss.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"score_threshold" : 0.7}
        )

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
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
import tempfile
import os

#LEITURA, E CRIAÇÃO DE CHUNKS
st.set_page_config(page_title="Agente Auxiliar de Estudos")
st.title("Agente de IA que ajuda nos estudos recebendo conteúdo por PDF")

uploaded_file = st.file_uploader("Envie um arquivo PDF para começar os estudos", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"PDF carregado com sucesso: {uploaded_file.name}")
        st.info(f"Número de páginas: {len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap = 50,
            separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        st.info(f"Total de chunks gerados: {len(chunks)}")

        with st.expander("Exemplo de chunk gerado"):
            st.write(chunks[0].page_content[:500] + ". . .")

    except Exception as e:
        st.error(f"Erro ao processar PDF {e}")

    finally:
     os.unlink(pdf_path)

#EMBEDDINGS E CRIAÇÃO DA VECTOR STORE
embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",
                                        model_kwargs={"trust_remote_code": True})

db_faiss = FAISS.from_documents(chunks, embedding_model)
db_faiss.save_local("vectorstore/faiss_index")

st.success("Embbeddings gerados e vector store salva com sucesso!")
st.info(f"Base vetorial criada com {len(chunks)} chunks vetorizados")

user_question = st.text_input("Faça suas perguntas!")

if st.button("Resposta") and user_question:
    db_faiss = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)

    retriever = db_faiss.as_retriever(search_type="similarity", search_kwargs={"k":3})
    relevant_documents = retriever.get_relevant_documents(user_question)

    for i, doc in enumerate(relevant_documents):
        print(f"\n--- Documento {i+1} ---")
        print(doc.page_content)

    context = "\n\n".join([doc.page_content for doc in relevant_documents])
    
    prompt = f"""
Responda de forma clara e didática com base no seguinte conteúdo extraído do PDF:

{context}

pergunta: {user_question}

resposta:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"llama3",
            "prompt":prompt,
            "stream":False
        }
    )

    model_response = response.json()["response"]
    st.markdown(f"**Resposta:** {model_response}")
import streamlit as st
import tempfile
from document_processor import DocumentProcessor
from memory_manager import MemoryManager
from llm_service import LLMService
from workflow_engine import WorkflowEngine, GraphState
from langchain.embeddings import HuggingFaceEmbeddings

class StudyAgentUI:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.llm_service = LLMService()
        self.doc_processor = DocumentProcessor()
        self.workflow = WorkflowEngine(self.memory_manager, self.llm_service)

    def run(self):
        st.title("Agente de Estudos!")
        upload_file = st.file_uploader("Envie um PDF", type=".pdf", key="pdf_uploader_unique")
        
        # Só processa o PDF se for um novo upload
        if upload_file is not None:
            if "last_uploaded_filename" not in st.session_state or \
               st.session_state.last_uploaded_filename != upload_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(upload_file.read())
                    pdf_path = tmp.name
                with st.spinner("Carregando e processando documento..."):
                    chunks = self.doc_processor.load_and_chunk(pdf_path)
                    retriever = self.doc_processor.create_retriever(chunks)
                st.session_state.retriever = retriever
                st.session_state.last_uploaded_filename = upload_file.name
                st.success("Documento processado com sucesso!")
        retriever = st.session_state.get("retriever", None)

        user_input = st.text_input("Digite sua pergunta ou peça por uma questão:")
        # Removido campo de resposta opcional
        if st.button("Executar") and user_input and retriever:
            flow = self.workflow.build_graph()
            initial_state: GraphState = {
                "pergunta": user_input,
                "retriever": retriever,
                # "resposta_usuario": user_answere,  # Removido
                "memoria": [],
                "resposta": "",
                # "avaliacao": ""  # Removido
            }
            resultado = flow.invoke(initial_state)
            if "resposta" in resultado and resultado["resposta"]:
                st.subheader("Resposta")
                st.write(resultado["resposta"])
            # Removido bloco de avaliação
        if st.button("🔍 Ver memória"):
            all_results = self.memory_manager.list_all_memories()
            if all_results:
                st.subheader(f"🧠 Conteúdo da memória ({len(all_results)} itens):")
                for r in all_results:
                    messages = r.get("messages", [])
                    if messages:
                        for msg in messages:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            st.markdown(f"- **{r.get('created_at', '')}** [{role}]: {content[:200]}...")
                    else:
                        content = r.get('message', r.get('memory', ''))
                        st.markdown(f"- **{r.get('created_at', '')}** ➜ {content[:200]}...")
            else:
                st.warning("Nenhum item encontrado na memória.") 
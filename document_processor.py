from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from config import EMBEDDING_MODEL, CHUNK_PERCENTILE, SEARCH_K

class DocumentProcessor:
    """Processa PDF, faz chunking e cria retriever"""
    def __init__(self):
        print("[DEBUG] Inicializando HuggingFaceEmbeddings...")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"trust_remote_code": True}
            )
            print("[DEBUG] Modelo de embeddings carregado com sucesso!")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar HuggingFaceEmbeddings: {e}")
            raise

    def load_and_chunk(self, pdf_path: str):
        print(f"[DEBUG] Carregando PDF: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[DEBUG] {len(documents)} documentos carregados.")
        splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=CHUNK_PERCENTILE
        )
        chunks = splitter.split_documents(documents)
        print(f"[DEBUG] {len(chunks)} chunks gerados.")
        return chunks

    def create_retriever(self, chunks):
        print(f"[DEBUG] Criando retriever com {len(chunks)} chunks...")
        db_faiss = FAISS.from_documents(chunks, self.embedding_model)
        print("[DEBUG] Retriever criado com sucesso!")
        return db_faiss.as_retriever(search_type="similarity", search_kwargs={"k": SEARCH_K}) 
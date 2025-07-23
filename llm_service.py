from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from config import LLM_REPO_ID, HF_TOKEN, LLM_MAX_TOKENS, LLM_DO_SAMPLE, LLM_REPETITION_PENALTY
from langchain_community.embeddings import HuggingFaceEmbeddings

class LLMService:
    """Serviço para interagir com o LLM via LangChain"""
    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            task="text-generation",
            max_new_tokens=LLM_MAX_TOKENS,
            do_sample=LLM_DO_SAMPLE,
            repetition_penalty=LLM_REPETITION_PENALTY,
            huggingfacehub_api_token=HF_TOKEN  # Certifique-se que HF_TOKEN está correto
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)
        # Removido o carregamento de HuggingFaceEmbeddings daqui

    def call(self, prompt: str):
        response = self.chat_model.invoke(prompt)
        return getattr(response, "content", str(response))

    def classify(self, question: str) -> str:
        prompt = (
            f"Você é um classificador de mensagens. Após analisar a seguinte pergunta: '{question}', "
            "responda apenas com: explicar, gerar_pergunta ou avaliar."
        )
        result = self.call(prompt).lower().strip()
        if "explicar" in result:
            return "explicar"
        elif "gerar_pergunta" in result:
            return "gerar_pergunta"
        elif "avaliar" in result:
            return "avaliar"
        return "explicar" 
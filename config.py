"""
Configurações centralizadas do Agente de Estudos
"""
import os
import requests
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv('./env')

# Tokens e chaves
MEM0_API_KEY = os.getenv("MEM0_API_KEY") 
HF_TOKEN = os.getenv("HF_TOKEN") 

# IDs fixos
MEMORY_ID = "agente_estudos"
USER_ID = "usuario1"

# Modelos
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/gemma-3-12b-it"

# Chunking
CHUNK_PERCENTILE = 90
SEARCH_K = 4

# LLM
LLM_MAX_TOKENS = 1024
LLM_DO_SAMPLE = False
LLM_REPETITION_PENALTY = 1.03

# Outras configs
MEMORY_DAYS_BACK = 7

class Config:
    """Classe de configuração centralizada"""
    
    # API Keys
    MEM0_API_KEY = "m0-bFj5UTXVlpNpSnlaDOfyrfTdYiAWQxcYiQbOHqxk"
    HF_TOKEN = os.getenv("HF_TOKEN") or "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"
    
    # Configurações do modelo de linguagem
    # Usar API mais simples e confiável
    MODEL_REPO_ID = "microsoft/DialoGPT-small"  # Tentar novamente
    MODEL_MAX_TOKENS = 128
    MODEL_REPETITION_PENALTY = 1.1

    # Modelo de fallback (ainda menor)
    FALLBACK_MODEL_REPO_ID = "gpt2"  # Tentar novamente
    FALLBACK_MAX_TOKENS = 64

    # Configuração para API alternativa
    USE_SIMPLE_RESPONSE = True  # Usar respostas simples se API falhar
    
    # Configurações de embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Configurações de memória
    MEMORY_ID = "agente_estudos"
    USER_ID = "usuario1"
    MEMORY_DESCRIPTION = "Memória do Agente de Estudos"
    
    # Configurações de busca
    SEARCH_K = 4
    MEMORY_DAYS_BACK = 7
    
    # Configurações de chunking
    CHUNK_PERCENTILE = 90
    
    # Configurações de timeout e conectividade
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    @classmethod
    def validate_tokens(cls):
        """Valida se os tokens necessários estão disponíveis"""
        if not cls.HF_TOKEN:
            print("Token não encontrado!")
            import getpass
            cls.HF_TOKEN = getpass.getpass("Por favor, digite seu token: ")
            os.environ["HF_TOKEN"] = cls.HF_TOKEN
    
    @classmethod
    def check_connectivity(cls):
        """Verifica conectividade com serviços externos"""
        try:
            # Testar conectividade com HuggingFace
            response = requests.get("https://huggingface.co", timeout=cls.REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("✅ Conectividade com HuggingFace: OK")
            else:
                print("⚠️ Conectividade com HuggingFace: Problema")
                return False
        except Exception as e:
            print(f"❌ Erro de conectividade com HuggingFace: {e}")
            return False
        
        try:
            # Testar conectividade com Mem0
            response = requests.get("https://api.mem0.ai", timeout=cls.REQUEST_TIMEOUT)
            if response.status_code in [200, 401, 403]:  # 401/403 são normais sem auth
                print("✅ Conectividade com Mem0: OK")
            else:
                print("⚠️ Conectividade com Mem0: Problema")
                return False
        except Exception as e:
            print(f"❌ Erro de conectividade com Mem0: {e}")
            return False
        
        return True 
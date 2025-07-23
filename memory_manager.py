"""
Gerenciador de memória para o Agente de Estudos
"""
from mem0 import MemoryClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from config import MEMORY_ID, USER_ID, MEM0_API_KEY, MEMORY_DAYS_BACK


class MemoryManager:
    """Gerencia operações de memória usando Mem0"""
    
    def __init__(self):
        self.client = MemoryClient(api_key=MEM0_API_KEY)
        self._initialize_memory()
    
    def _initialize_memory(self):
        try:
            self.client.create(memory_id=MEMORY_ID, user_id=USER_ID, description="Memória do Agente de Estudos")
        except Exception:
            pass  # Memória já existe
    
    def save_conversation(self, user_message: str, assistant_message: str) -> bool:
        """Salva uma conversa na memória"""
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
            
            self.client.add(
                messages=messages,
                user_id=USER_ID,
                memory_id=MEMORY_ID
            )
            return True
        except Exception as e:
            return False
    
    def search_relevant_memories(self, query: str) -> str:
        """Busca memórias relevantes para uma query"""
        try:
            results = self.client.search(
                query=query,
                user_id=USER_ID,
                memory_id=MEMORY_ID
            )
            
            # Filtrar memórias recentes
            last_week = datetime.now(timezone.utc) - timedelta(days=MEMORY_DAYS_BACK)
            recent = [
                r for r in results 
                if datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc) >= last_week
            ]
            
            return "\n".join([r.get("message", r.get("memory", "")) for r in recent])
        except Exception as e:
            return ""
    
    def list_all_memories(self) -> List[Dict[str, Any]]:
        """Lista todas as memórias usando diferentes estratégias"""
        try:
            # Estratégia 1: Query vazia
            try:
                results = self.client.search(
                    query=" ",
                    user_id=USER_ID,
                    memory_id=MEMORY_ID
                )
                if results:
                    return results
            except:
                pass
            
            # Estratégia 2: Query genérica
            try:
                results = self.client.search(
                    query="a",
                    user_id=USER_ID,
                    memory_id=MEMORY_ID
                )
                if results:
                    return results
            except:
                pass
            
            # Estratégia 3: Buscar por termos comuns
            common_terms = ["pergunta", "resposta", "explicação", "avaliação", "conclusão"]
            all_results = []
            
            for term in common_terms:
                try:
                    results = self.client.search(
                        query=term,
                        user_id=USER_ID,
                        memory_id=MEMORY_ID
                    )
                    all_results.extend(results)
                except:
                    continue
            
            # Remover duplicatas
            unique_results = []
            seen_ids = set()
            for r in all_results:
                if r.get('id') not in seen_ids:
                    unique_results.append(r)
                    seen_ids.add(r.get('id'))
            return unique_results
            
        except Exception as e:
            return [] 
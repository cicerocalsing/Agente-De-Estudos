from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, List, Dict
from memory_manager import MemoryManager
from llm_service import LLMService
from langdetect import detect
import re

class GraphState(TypedDict):
    pergunta: str
    retriever: Any
    resposta_usuario: str
    memoria: List[Any]
    resposta: str
    avaliacao: str

class WorkflowEngine:
    def __init__(self, memory_manager: MemoryManager, llm_service: LLMService):
        self.memory_manager = memory_manager
        self.llm_service = llm_service

    def route_input(self, state: GraphState) -> str:
        question = state["pergunta"]
        return self.llm_service.classify(question)

    def explanation_node(self, state: GraphState) -> Dict:
        question = state["pergunta"]
        docs = state["retriever"].get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        history = self.memory_manager.search_relevant_memories(question)
        idioma = detect(question)
        if idioma == "en":
            idioma_instrucao = "Please answer in English."
        elif idioma == "pt":
            idioma_instrucao = "Por favor, responda em português."
        else:
            idioma_instrucao = "Please answer in the same language as the question."
        prompt = f"""Você é um assistente educacional inteligente e prestativo da plataforma Cicero Tech AI, criado para ajudar estudantes a entender conteúdos acadêmicos com clareza, profundidade e foco. Sua missão é gerar explicações completas e acessíveis, respeitando o nível do usuário e sempre utilizando o conteúdo presente no material de estudo.

---
📚 Instruções de comportamento:
- Seja claro, direto e didático.  
- Nunca invente informações que não estejam no contexto ou histórico.  
- Se não houver informações suficientes, indique isso educadamente e oriente o usuário a revisar o material.  
- Utilize exemplos sempre que possível.  
- Evite respostas genéricas ou vagas.  
- Não inclua frases como “sou um assistente” ou “sou uma IA”, apenas entregue o conteúdo como um tutor humano faria.  
- Responda a pergunta do usuário no idioma que ela foi feita.
- {idioma_instrucao}

---
🧠 Contexto do conteúdo extraído do PDF:  
{context}

📜 Histórico recente de interações relevantes:  
{history}

❓ Pergunta feita pelo usuário:  
{question}

---
🔍 Agora, com base no conteúdo e no histórico, explique a resposta à pergunta acima de forma clara, precisa e amigável. Use exemplos se forem úteis."""

        answer = self.llm_service.call(prompt)
        self.memory_manager.save_conversation(question, answer)
        return {"resposta": answer}

    def question_generation_node(self, state: GraphState) -> Dict:
        docs = state["retriever"].get_relevant_documents("Conteúdo Importante!")
        context = "\n".join([doc.page_content for doc in docs])
        history = self.memory_manager.search_relevant_memories("gerar perguntas")
        user_question = state["pergunta"] if "pergunta" in state else ""
        idioma = detect(user_question) if user_question else "pt"
        # Detectar quantidade de perguntas
        match = re.search(r"(\d+)\s+pergunt", user_question.lower())
        n_perguntas = int(match.group(1)) if match else 1
        if idioma == "en":
            idioma_instrucao = f"Please generate {n_perguntas} multiple-choice question(s) and all alternatives in English."
        elif idioma == "pt":
            idioma_instrucao = f"Por favor, gere {n_perguntas} pergunta(s) de múltipla escolha e todas as alternativas em português."
        else:
            idioma_instrucao = f"Please generate {n_perguntas} multiple-choice question(s) and all alternatives in the same language as the user's request."
        exemplo_formato = f"""
Exemplo de formato:
1. [Pergunta 1]
a) ...
b) ...
c) ...
d) ...
Gabarito: Letra X - [resposta correta]

2. [Pergunta 2]
a) ...
b) ...
c) ...
d) ...
Gabarito: Letra X - [resposta correta]
"""
        prompt = f"""Você é um assistente educacional inteligente da plataforma Cicero Tech AI. Sua tarefa é gerar **perguntas de múltipla escolha** sobre conteúdos acadêmicos com clareza, coerência e foco pedagógico. As perguntas devem ajudar o estudante a revisar e testar seu entendimento sobre o conteúdo apresentado abaixo.

---
🎯 Instruções de comportamento:
- Gere exatamente {n_perguntas} perguntas, numeradas (1., 2., 3., ...), cada uma com 4 alternativas (a, b, c, d) e o gabarito ao final de cada pergunta.
- Separe cada pergunta com uma linha em branco.
- As perguntas devem ser objetivas, claras e diretamente relacionadas ao conteúdo.
- Se o conteúdo estiver genérico, crie perguntas abrangentes e coerentes.
- Não invente informações fora do conteúdo apresentado.
- Use linguagem acessível e pedagógica.
- Não se identifique como assistente ou IA.
- {idioma_instrucao}
- {exemplo_formato}

---
📚 Contexto do conteúdo extraído do PDF:  
{context}

🧠 Histórico recente de interações com o usuário:  
{history}

---
✍️ Com base no conteúdo acima, crie {n_perguntas} pergunta(s) de múltipla escolha com 4 opções cada.  
Inclua o gabarito ao final de cada pergunta no formato:  
**Gabarito: Letra X - [resposta correta]**
"""
        question = self.llm_service.call(prompt)
        self.memory_manager.save_conversation("Me pergunte algo", question)
        return {"resposta": question}

    def build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("explicar", self.explanation_node)
        graph.add_node("gerar_pergunta", self.question_generation_node)
        graph.set_conditional_entry_point(self.route_input)
        graph.add_edge("explicar", END)
        graph.add_edge("gerar_pergunta", END)
        return graph.compile() 
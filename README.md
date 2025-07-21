# 🧠 Agente de Estudos

Um agente inteligente para análise, explicação, geração de questões e avaliação de respostas sobre conteúdos de arquivos PDF, utilizando IA, LangChain, LangGraph, HuggingFace e Mem0.

---

## 🎯 Objetivo

O **Agente de Estudos** foi criado para:
- Ajudar estudantes a entender conteúdos acadêmicos de PDFs com explicações claras e didáticas.
- Gerar questões de múltipla escolha baseadas no conteúdo do material.
- Avaliar respostas fornecidas pelo usuário, com feedback construtivo.
- Manter um histórico de interações e respostas em memória persistente.

---

## 🚀 Funcionalidades

- **Upload de PDF:** Carregue qualquer arquivo PDF para análise.
- **Chunking e Indexação:** O conteúdo é dividido em partes semânticas e indexado para busca eficiente.
- **Perguntas e Respostas:** Faça perguntas sobre o conteúdo e receba explicações detalhadas.
- **Geração de Questões:** Gere automaticamente questões de múltipla escolha sobre o material.
- **Avaliação de Respostas:** Receba avaliação e feedback sobre respostas fornecidas.
- **Memória Persistente:** Todo o histórico de interações é salvo e pode ser consultado.
- **Interface Web:** Interface amigável via Streamlit.

---

## ⚙️ Como Usar

1. **Instale as dependências:**
   ```bash
   pip install streamlit langchain langchain-community langchain-huggingface langgraph mem0 python-dotenv
   ```

2. **Configure as variáveis de ambiente:**
   - Crie um arquivo `.env` (opcional) ou edite no início do `main.py`:
     ```
     HF_TOKEN=seu_token_huggingface
     MEM0_API_KEY=sua_chave_mem0
     ```
   - Ou edite diretamente no código as linhas:
     ```python
     os.environ['MEM0_API_KEY'] = "SUA_CHAVE"
     os.environ['HF_TOKEN'] = "SEU_TOKEN"
     ```

3. **Execute a aplicação:**
   ```bash
   streamlit run main.py
   ```

4. **No navegador:**
   - Faça upload de um PDF.
   - Digite sua pergunta ou solicite uma questão.
   - (Opcional) Escreva uma resposta para ser avaliada.
   - Clique em **Executar**.
   - Veja a resposta, avaliação e consulte a memória das interações.

---

## 🧩 Principais Dependências

- **[Streamlit](https://streamlit.io/):** Interface web.
- **[LangChain](https://python.langchain.com/):** Orquestração de LLMs e retrievers.
- **[LangGraph](https://github.com/langchain-ai/langgraph):** Workflow dinâmico com grafos de estados.
- **[HuggingFace Hub](https://huggingface.co/):** Modelos de linguagem (ex: `google/gemma-3-12b-it`).
- **[Mem0](https://mem0.ai/):** Memória persistente de interações.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** Gerenciamento de variáveis de ambiente.

---

## 📚 Estrutura do Código

- **main.py** — Toda a lógica do agente, interface, workflow, memória e integração com LLM.
- **Dependências** — Instaladas via pip (veja acima).

---

## 📝 Observações e Pré-requisitos

- **Token HuggingFace:** Necessário para acessar modelos LLM via API. Recomenda-se um token PRO para modelos grandes.
- **Chave Mem0:** Necessária para salvar e buscar memórias.
- **Modelos:** O código usa por padrão o modelo `google/gemma-3-12b-it`. Se não tiver acesso, troque por um modelo público (ex: `HuggingFaceH4/zephyr-7b-beta`).
- **Python 3.8+** recomendado.
- **Ambiente virtual** recomendado para evitar conflitos de dependências.

---

## 🏗️ Extensões Possíveis

- Suporte a outros tipos de arquivos (docx, txt, etc).
- Integração com outros LLMs (OpenAI, Cohere, etc).
- Interface multiusuário.
- Exportação de histórico/memória.

---

## 💡 Exemplo de Uso

1. Faça upload de um PDF acadêmico.
2. Pergunte: “Qual o tema principal do PDF?”
3. Peça: “Gere uma questão de múltipla escolha sobre o conteúdo.”
4. Escreva uma resposta e peça avaliação.
5. Consulte o histórico de interações.

---

## 🛡️ Licença

Uso educacional e experimental. Para uso comercial, consulte as licenças dos modelos e APIs utilizadas. 
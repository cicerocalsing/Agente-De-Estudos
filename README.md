
# 🧠 Study Agent

An intelligent agent for analyzing, explaining, generating questions, and evaluating answers from PDF file content using AI, LangChain, LangGraph, HuggingFace, and Mem0.

---

## 🎯 Purpose

The **Study Agent** was built to:
- Help students understand academic PDF content with clear, didactic explanations.
- Generate multiple-choice questions based on study material.
- Evaluate user-provided answers with constructive feedback.
- Maintain a persistent memory of interactions and responses.

---

## 🚀 Features

- **PDF Upload:** Upload any PDF file for analysis.
- **Chunking & Indexing:** Content is semantically split and indexed for efficient retrieval.
- **Q&A:** Ask questions about the content and get detailed explanations.
- **Question Generation:** Automatically generate multiple-choice questions based on the material.
- **Answer Evaluation:** Receive feedback and assessment on your answers.
- **Persistent Memory:** All interactions are saved and can be reviewed.
- **Web Interface:** Friendly UI via Streamlit.

---

## ⚙️ How to Use

1. **Install dependencies:**
   ```bash
   pip install streamlit langchain langchain-community langchain-huggingface langgraph mem0 python-dotenv
   ```

2. **Set environment variables:**
   - Create a `.env` file (optional) or edit at the top of `main.py`:
     ```
     HF_TOKEN=your_huggingface_token
     MEM0_API_KEY=your_mem0_key
     ```
   - Or directly edit in the code:
     ```python
     os.environ['MEM0_API_KEY'] = "YOUR_KEY"
     os.environ['HF_TOKEN'] = "YOUR_TOKEN"
     ```

3. **Run the app:**
   ```bash
   streamlit run main.py
   ```

4. **In the browser:**
   - Upload a PDF.
   - Type a question or request a generated one.
   - (Optional) Provide an answer for evaluation.
   - Click **Run**.
   - View the explanation, evaluation, and memory log.

---

## 🧩 Main Dependencies

- **[Streamlit](https://streamlit.io/):** Web interface.
- **[LangChain](https://python.langchain.com/):** LLM and retriever orchestration.
- **[LangGraph](https://github.com/langchain-ai/langgraph):** Dynamic state graph workflow.
- **[HuggingFace Hub](https://huggingface.co/):** LLMs (e.g., `google/gemma-3-12b-it`).
- **[Mem0](https://mem0.ai/):** Persistent memory storage.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** Environment variable management.

---

## 📚 Code Structure

- **main.py** — Core logic for the agent, UI, workflow, memory, and LLM integration.
- **Dependencies** — Installed via pip (see above).

---

## 📝 Notes & Requirements

- **HuggingFace Token:** Required to access LLMs via API. A PRO token is recommended for large models.
- **Mem0 Key:** Required for storing and retrieving memory.
- **Models:** Defaults to `google/gemma-3-12b-it`. If unavailable, try a public model like `HuggingFaceH4/zephyr-7b-beta`.
- **Python 3.8+** recommended.
- **Virtual environment** is recommended to avoid dependency conflicts.

---

## 🏗️ Possible Extensions

- Support for other file types (docx, txt, etc.).
- Integration with other LLMs (OpenAI, Cohere, etc.).
- Multi-user interface.
- Export of memory/history.

---

## 💡 Example Usage

1. Upload an academic PDF.
2. Ask: “What’s the main topic of this PDF?”
3. Request: “Generate a multiple-choice question based on the content.”
4. Write an answer and request evaluation.
5. Check interaction history.

---

## 🛡️ License

Educational and experimental use. For commercial use, consult the licenses of the models and APIs used.

# Chat Bot supports learnings

## 1. Introduction
A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering interactions with PDF documents. The system uses local LLM (Mistral) combined with vector similarity
search to provide accurate, context-aware responses based on the content of uploaded PDF documents.

## 2. Features
- PDF document processing and text extraction
- Intelligent text chunking for optimal context retrieval
- Vector similarity search using FAISS
- Local LLM integration for private and cost-effective processing
- Context-aware responses with source tracking
- Support for Vietnamese language

## 3. Architecture
![...}(https://github.com/tranvietcuong03/Chatbot-supports-learning/blob/master/Image/rag.png)

## 4. Installation
- Technologies
```sh
  pip install -r requirements.txt
  ```
Need to download **Ollama** (Link here: ![ollama](https://www.ollama.com/) )

## 5. Setup
1. Open cmd, run:
 ```sh
  ollama serve
  ```
2. Create vectorstories (faiss folder):
```sh
  python populate_db.py
  ```
3. Run application:
   ```sh
  python app.py
  ```

## 6. Demo
- Demo the interface:
![...](https://github.com/tranvietcuong03/Chatbot-supports-learning/blob/master/Image/demo.png)

- Pytest: one of tests:
![...](https://github.com/tranvietcuong03/Chatbot-supports-learning/blob/master/Image/pytest_ex.png)

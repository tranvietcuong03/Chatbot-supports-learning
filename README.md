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
![...](https://github.com/tranvietcuong03/Chatbot-supports-learning/blob/master/Image/rag.png)

## 4. Technologies Used
- Python 3.10+
- LangChain framework
- FAISS for vector similarity search
- Mistral LLM
- PyPDF for document processing
- Vector embeddings
- Flasks

## 4. Installation
- Technologies
```sh
  pip install -r requirements.txt
  ```
Need to download **Ollama** (Link here: [ollama](https://www.ollama.com/) )

## 5. Setup
1. Open cmd, run:
 ```sh
  ollama serve
  ```
2. Create vectorstories (faiss folder):
```bash
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

## 7. Prompt
As a Vietnamese chatbot, I adjust the prompt template (reference from [vinallama](https://huggingface.co/vilm/vinallama-7b-chat-GGUF) ), there is the prompt template i use in this project: <br>
```
PROMPT_TEMPLATE = """<|im_start|>system
Bạn là một trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là đọc hiểu và trả lời câu hỏi dựa trên nội dung tài liệu được cung cấp.

Ngữ cảnh từ tài liệu:
{context}

Yêu cầu:
1. Trả lời hoàn toàn bằng tiếng Việt
2. Trả lời chi tiết, đầy đủ với các luận điểm và ví dụ từ tài liệu
3. Nếu không tìm thấy thông tin trong ngữ cảnh, chỉ trả lời "Câu hỏi này không có trong tài liệu"
4. Không được tự tạo thông tin không có trong ngữ cảnh  

Câu hỏi: {question}
<|im_end|>

<|im_start|>user
{question}
<|im_end|>

<|im_start|>assistant
"""
```

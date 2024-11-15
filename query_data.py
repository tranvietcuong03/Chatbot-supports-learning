import argparse
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

import warnings
warnings.filterwarnings('ignore')

FAISS_PATH = "faiss"

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


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Câu hỏi")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(
        model="mistral", 
        temperature=0.2,
        top_k=30
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Câu trả lời: {response_text}\nNguồn tham khảo: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
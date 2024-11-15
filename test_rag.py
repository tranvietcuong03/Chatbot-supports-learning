from query_data import query_rag
from langchain_community.llms import Ollama
import pytest

import warnings
warnings.filterwarnings('ignore')


EVAL_PROMPT = """<|im_start|>system
Bạn là một trợ lý AI đánh giá độ chính xác của câu trả lời. Nhiệm vụ của bạn là so sánh câu trả lời thực tế với câu trả lời mong đợi và đánh giá xem chúng có tương đồng về mặt ngữ nghĩa hay không.

Câu trả lời mong đợi: {expected_response}
Câu trả lời thực tế: {actual_response}

Yêu cầu:
1. Chỉ trả lời "Đúng" hoặc "Sai"
2. Trả lời "Đúng" nếu hai câu trả lời mang cùng ý nghĩa, bỏ qua sự khác biệt nhỏ về cách diễn đạt
3. Trả lời "Sai" nếu hai câu trả lời khác nhau về mặt ý nghĩa
4. Nếu một trong hai câu trả lời là "Câu hỏi này không có trong tài liệu", chỉ trả lời Đúng khi câu còn lại cũng thể hiện điều tương tự
<|im_end|>

<|im_start|>assistant
"""

def get_test_cases():
    return [
        {
            "question": "Triết học là gì?",
            "expected": """Triết học nghiên cứu thế giới một cách chỉnh thể, tìm ra những quy luật chung nhất chi phối sự vận động 
                của chỉnh thể đó nói chung, của xã hội loài người, của con người trong cuộc sống cộng đồng nói riêng và thể hiện nó một cách có hệ thống dưới dạng duy lý"""
        },
        {
            "question": "Hãy nêu khái niệm sản xuất vật chất",
            "expected": """Sản xuất vật chất là quá trình con người sử dụng công cụ lao động tác động vào tự nhiên, 
                cải biến các dạng vật chất của giới tự nhiên nhằm tạo ra của cải vật chất thoả mãn nhu cầu tồn tại 
                và phát triển của con người."""
        },
        {
            "question": "Định nghĩa giai cấp",
            "expected": """Người ta gọi giai cấp, những tập đoàn to lớn gồm những người khác nhau về địa vị của họ 
                trong một hệ thống sản xuất xã hội nhất định trong lịch sử, khác nhau về quan hệ của họ (thường thường thì những quan hệ này được pháp luật quy định và 
                thừa nhận đối với những tư liệu sản thường thì những quan hệ này được pháp luật quy định và thừa  nhận đối với những  tư liệu sản 
                xuất, về vai trò của họ trong tổ chức lao động xã hội, và như vậy là khác nhau về cách thức hưởng 
                thụ và về phần của cải xã hội ít hoặc nhiều mà họ được hưởng. Giai cấp là những tập đoàn người mà tập đoàn này có thể chiếm đoạt lao động của tập đoàn khác, 
                do chỗ các tập đoàn có địa vị khác nhau trong một chế độ kinh tế - xã hội nhất định"""
        }
        {
            "question": "Chủ nghĩa duy vật là gì?",
            "expected": """Là kết quả nhận thức của các nhà sáng lập chủ nghĩa Mác. Mác, Ăngghen, Lênin đã kế thừa 
                        những tinh hoa của các học thuyết trước đó, đồng thời khắc phục những hạn chế, sai lầm của chủ 
                        nghĩa duy vật siêu hình, dựa trên những thành tựu của khoa học hiện đại đã sáng lập ra chủ nghĩa 
                        duy vật biện chứng. Chủ nghĩa duy vật biện chứng của triết học Mác Lênin mang tính chất cách 
                        mạng triệt để và biện chứng khoa học, không chỉ phản ánh hiện thực đúng như bản thân nó mà  
                        còn là công cụ hữu ích giúp con người cải tạo hiện thực đó"""
        }
        # Add more
    ]

def query_and_validate(question, expected_response):
    actual_response = query_rag(question)
    
    print("\nQuestion:", question)
    print("Expected:", expected_response)
    print("Actual:", actual_response)
    
    eval_input = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=actual_response
    )
    
    llm = Ollama(model="mistral") 
    result = llm.invoke(eval_input).strip().lower()
    
    is_valid = result == "đúng"
    
    if not is_valid:
        print("❌ Test failed!")
        print(f"Evaluation result: {result}")
    else:
        print("✅ Test passed!")
        print(f"Evaluation result: {result}")
        
    return is_valid

@pytest.mark.parametrize("test_case", get_test_cases())
def test_rag_responses(test_case):
    assert query_and_validate(
        question=test_case["question"],
        expected_response=test_case["expected"]
    ), f"Response did not match expectations for question: {test_case['question']}"

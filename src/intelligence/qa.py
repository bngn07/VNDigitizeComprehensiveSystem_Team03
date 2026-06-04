import google.generativeai as genai


class QAEngine:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def answer(self, question, context):
        prompt = f"""
        Bạn là trợ lý phân tích tài liệu pháp lý.
        Chỉ được trả lời dựa trên nội dung dưới đây.
        ====================
        TÀI LIỆU
        ====================
        {context}
        ====================
        CÂU HỎI
        ====================
        {question}

        Nếu không tìm thấy thông tin thì trả lời: "Không tìm thấy thông tin trong tài liệu."
        """
        response = self.model.generate_content(prompt)
        return response.text
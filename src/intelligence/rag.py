from __future__ import annotations
from .chunker import Chunker
from .qa import QAEngine

class RAGPipeline:

    def __init__(self):
        self.chunker = Chunker()
        self.qa = QAEngine(
            # api_key="GEMINI_API_KEY" # Thay bằng API key thực tế
        )
        self.document_chunks = []

    def build(self, ocr_result):
        """
        Xây dựng danh sách chunk. 
        Hỗ trợ linh hoạt cả Object OCRResult (gốc của nhóm) lẫn chuỗi văn bản thô.
        """
        # Trường hợp 1: Nếu đầu vào là chuỗi str 
        if isinstance(ocr_result, str):
            chunk_size = 1000
            self.document_chunks = [ocr_result[i:i + chunk_size] for i in range(0, len(ocr_result), chunk_size)]
        
        # Trường hợp 2: Đầu vào chuẩn là Object OCRResult (Có thuộc tính .blocks)
        else:
            try:
                # Gọi hàm chunker gốc của nhóm dựa trên cấu trúc blocks
                self.document_chunks = self.chunker.chunk(ocr_result)
            except AttributeError:
                # Phòng hờ nếu object truyền vào không đúng cấu trúc blocks mong muốn
                # Thử ép chuyển đổi text thô nếu hàm chunker có cơ chế fallback
                ocr_text = "\n".join([block.text for block in ocr_result.blocks]) if hasattr(ocr_result, 'blocks') else str(ocr_result)
                chunk_size = 1000
                self.document_chunks = [ocr_text[i:i + chunk_size] for i in range(0, len(ocr_text), chunk_size)]

    def ask(self, question):
        if not self.document_chunks:
            return "Không tìm thấy dữ liệu ngữ cảnh từ tài liệu để trả lời."

        # Lấy tối đa 5 đoạn ngữ cảnh đầu tiên khớp với tài liệu số hóa để đóng gói vào Prompt gửi API
        context = "\n".join(self.document_chunks[:5])

        return self.qa.answer(question, context)
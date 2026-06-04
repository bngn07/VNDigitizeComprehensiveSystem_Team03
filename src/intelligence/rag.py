# from __future__ import annotations
# from typing import Any # Dùng tạm Any nếu chưa import OCRResult chính thức
# from .chunker import build_chunks_from_ocr
# from .embedder import Embedder
# from .vector_store import VectorStore
# from .retriever import Retriever
# from .qa import QAEngine


# class RAGPipeline:
#     def __init__(self, chunk_size: int = 500, overlap: int = 100):
#         self.chunk_size = chunk_size
#         self.overlap = overlap

#         # Khởi tạo Embedder (đã được bọc cấu hình CPU an toàn)
#         self.embedder = Embedder()

#         self.vector_store: VectorStore | None = None
#         self.retriever: Retriever | None = None
#         self.qa_engine = QAEngine()

#     def build(self, ocr_data: Any) -> None:
#         """
#         Xây dựng cơ sở dữ liệu vector từ kết quả OCR (Chấp nhận cả OCRResult hoặc chuỗi Str)
#         """
#         # 💡 Kiểm tra đầu vào: Nếu main.py truyền vào chuỗi văn bản đã join
#         if isinstance(ocr_data, str):
#             # Nếu build_chunks_from_ocr chỉ nhận OCRResult, ta tự cắt chunk bằng str đơn giản ở đây để backup
#             if hasattr(build_chunks_from_ocr, '__code__'):
#                 # Giả định hàm chunker của bạn xử lý text thô hoặc cấu trúc riêng
#                 try:
#                     chunks = build_chunks_from_ocr(ocr_data, chunk_size=self.chunk_size, overlap=self.overlap)
#                 except Exception:
#                     # Dự phòng nếu hàm chunker cũ bắt buộc phải nhận object phức tạp
#                     chunks = [ocr_data[i:i+self.chunk_size] for i in range(0, len(ocr_data), self.chunk_size - self.overlap)]
#         else:
#             chunks = build_chunks_from_ocr(ocr_data, chunk_size=self.chunk_size, overlap=self.overlap)

#         if not chunks:
#             raise ValueError("No chunks generated")

#         # Embedding tạo vector trên CPU
#         embeddings = self.embedder.encode(chunks)

#         # Khởi tạo và lưu trữ Vector Store
#         embedding_dim = embeddings.shape[1]
#         self.vector_store = VectorStore(embedding_dim=embedding_dim)
#         self.vector_store.add(embeddings=embeddings, texts=chunks)

#         # Kết nối Retriever
#         self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)

#     def ask(self, query: str, top_k: int = 3) -> str:
#         if self.retriever is None:
#             raise RuntimeError("RAGPipeline has not been built")

#         contexts = self.retriever.retrieve(query=query, top_k=top_k)
#         answer = self.qa_engine.answer(query=query, contexts=contexts)
#         return answer

from __future__ import annotations
from .chunker import Chunker
from .qa import QAEngine

class RAGPipeline:

    def __init__(self):
        self.chunker = Chunker()
        self.qa = QAEngine(
            api_key="AQ.Ab8RN6LD_p7JeSBGR2mtdCYhRXIP_Bb7WNzyxSgF2tCVGV9-AA"
        )
        self.document_chunks = []

    def build(self, ocr_result):
        """
        Xây dựng danh sách chunk. 
        Hỗ trợ linh hoạt cả Object OCRResult (gốc của nhóm) lẫn chuỗi văn bản thô.
        """
        # Trường hợp 1: Nếu đầu vào là chuỗi str (do main.py truyền full_text đã join)
        if isinstance(ocr_result, str):
            # Tự động cắt chuỗi theo độ dài ký tự để không làm sập hàm chunker gốc
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

        return self.qa.answer(
            question,
            context
        )
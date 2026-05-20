from symspellpy import SymSpell, Verbosity
import sys, re
from pathlib import Path
from typing import List, Tuple

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from .correction_type import Correction


class DictionaryCorrector:
    """Sửa lỗi sử dụng từ điển"""
    
    def __init__(self, confidence_threshold=0.8):
        self.sym_spell = None
        self.confidence_threshold = confidence_threshold
        self._load_dictionary()
        

    def _load_dictionary(self):
        try:
            from symspellpy import SymSpell, Verbosity
            self.Verbosity = Verbosity
            self.sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
            
            # Thử load từ điển tiếng Việt (bạn cần chuẩn bị file này)
            dict_path = root_dir / "dictionary" / "vi_VN.txt"
            if dict_path.exists():
                self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1, encoding="utf-8")
                print(f"Đã load từ điển tiếng Việt từ: {dict_path}")
            else:
                print("Chưa tìm thấy từ điển vi_VN.txt. DictionaryCorrector sẽ dùng mode cơ bản.")
                
        except ImportError:
            print(" symspellpy chưa được cài. Cài bằng lệnh: pip install symspellpy")
            self.sym_spell = None

    import re

    def correct_word(self, word: str) -> str:

        if self.sym_spell is None:
            return word

        word = word.strip()

        if not word:
            return word

    # Không sửa số
        if word.isdigit():
            return word

    # Không sửa từ quá ngắn
        if len(word) <= 2:
            return word

    # Không sửa acronym
        if word.isupper():
            return word

    # Không sửa token chứa ký tự đặc biệt
        if re.search(r'[/:-]', word):
            return word

        suggestions = self.sym_spell.lookup(
            word,
            self.Verbosity.CLOSEST,
            max_edit_distance=1
        )

        if suggestions:

            best = suggestions[0]

        # Chỉ sửa nếu khoảng cách edit nhỏ
            if best.distance <= 1:
                return best.term

        return word


    def correct(self, texts: List[str], confidences: List[float] = None) -> Tuple[List[str], List[Correction]]:

        if self.sym_spell is None:
            return texts.copy(), []

        corrected = texts.copy()
        corrections: List[Correction] = []

        for i, text in enumerate(texts):

            # Nếu confidence cao -> bỏ qua
            if confidences and confidences[i] >= self.confidence_threshold:
                continue

            words = text.split()
            corrected_words = []

            for word in words:

                corrected_word = self.correct_word(word)

                corrected_words.append(corrected_word)

                if corrected_word != word:
                    corrections.append(
                        Correction(
                        original_text=word,
                        corrected_text=corrected_word,
                        confidence=0.0,
                        position=i,
                        reason="dictionary"
                        )   
                    )

            corrected[i] = " ".join(corrected_words)

        return corrected, corrections
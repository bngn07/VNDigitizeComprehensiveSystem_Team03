from symspellpy import SymSpell, Verbosity
import sys
from pathlib import Path
from typing import List, Tuple

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from .correction_type import Correction


class DictionaryCorrector:
    """Sửa lỗi sử dụng từ điển"""
    
    def __init__(self):
        self.sym_spell = None
        self._load_dictionary()

    def _load_dictionary(self):
        try:
            from symspellpy import SymSpell, Verbosity
            self.Verbosity = Verbosity
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Thử load từ điển tiếng Việt (bạn cần chuẩn bị file này)
            dict_path = root_dir / "dictionary" / "vi_VN.txt"
            if dict_path.exists():
                self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
                print(f"Đã load từ điển tiếng Việt từ: {dict_path}")
            else:
                print("Chưa tìm thấy từ điển vi_VN.txt. DictionaryCorrector sẽ dùng mode cơ bản.")
                
        except ImportError:
            print(" symspellpy chưa được cài. Cài bằng lệnh: pip install symspellpy")
            self.sym_spell = None

    def correct(self, texts: List[str]) -> Tuple[List[str], List[Correction]]:
        if self.sym_spell is None:
            return texts.copy(), []  # Không làm gì nếu chưa có symspell

        corrected = texts.copy()
        corrections: List[Correction] = []

        for i, word in enumerate(texts):
            if len(word.strip()) < 2:
                continue

            # Tìm gợi ý gần nhất
            suggestions = self.sym_spell.lookup(
                word, 
                self.Verbosity.CLOSEST, 
                max_edit_distance=2
            )

            if suggestions and suggestions[0].term != word:
                new_word = suggestions[0].term
                corrected[i] = new_word
                
                corrections.append(Correction(
                    original_text=word,
                    corrected_text=new_word,
                    confidence=0.0,
                    position=i,
                    reason="dictionary"
                ))

        return corrected, corrections
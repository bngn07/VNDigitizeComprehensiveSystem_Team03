from symspellpy import SymSpell, Verbosity
import sys, re
from pathlib import Path
from typing import List, Tuple
from .ngram import BigramLanguageModel 
from unidecode import unidecode
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from .correction_type import Correction

OCR_NORMALIZATION = {
    "0": "o",
    "1": "l",
    "4": "a",
    "5": "s",
    "8": "b"
}

class DictionaryCorrector:
    
    def __init__(self, confidence_threshold=0.8):
        self.sym_spell = None
        self.confidence_threshold = confidence_threshold
        self.ngram_model = BigramLanguageModel()

        current_file = Path(__file__).resolve()
        root_dir = current_file.parents[4]
        corpus_path = root_dir / "dictionary" / "corpus.txt"

        if corpus_path.exists():
            self.ngram_model.load_corpus(str(corpus_path))
        else:
            print("Không tìm thấy corpus.txt")
        
        self._load_dictionary()
        
    def remove_accent(self, text: str) -> str:
        return unidecode(text).lower()
    
    def _load_dictionary(self):
        try:
            from symspellpy import SymSpell, Verbosity
            self.Verbosity = Verbosity
            self.sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
            
            current_file = Path(__file__).resolve()
            root_dir = current_file.parents[4]
            dict_path = root_dir / "dictionary" / "vi_VN.txt"
            
            if dict_path.exists():
                self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1, encoding="utf-8")
            else:
                print("Chưa tìm thấy từ điển vi_VN.txt. DictionaryCorrector sẽ dùng mode cơ bản.")
                
        except ImportError:
            print(" symspellpy chưa được cài. Cài bằng lệnh: pip install symspellpy")
            self.sym_spell = None
    
    def normalize_ocr_token(self, word: str) -> str:
        normalized = word.lower()
        for wrong, correct in OCR_NORMALIZATION.items():
            normalized = normalized.replace(wrong, correct)
        return normalized

    def correct_word(self, word: str, next_word: str = "") -> str:

        if self.sym_spell is None:
            return word
        original_word = word
        word = word.strip()

        if not word:
            return word

        if word.isdigit():
            return word

        if len(word) <= 2:
            return word

        if word.isupper():
            return word

        if re.search(r'[/:-]', word):
            return word
        
        normalized_word = self.normalize_ocr_token(word)
        suggestions = self.sym_spell.lookup(normalized_word, self.Verbosity.CLOSEST, max_edit_distance=1)
        if not suggestions:
            return word

        candidates = []
        for s in suggestions:
            candidate_no_accent = self.remove_accent(s.term)
            if candidate_no_accent == normalized_word:
                candidates.append(s.term)
        if not candidates:
                return word
        
        if not next_word:
            if candidates:
                return candidates[0]
            return word
        
        normalized_next_word = self.normalize_ocr_token(next_word)
        
        best_candidate = self.ngram_model.rank_candidates(next_word=normalized_next_word, candidates=candidates)

        if best_candidate and best_candidate != word:
            return best_candidate

        return word


    def correct(self, texts: List[str], confidences: List[float] = None) -> Tuple[List[str], List[Correction]]:

        if self.sym_spell is None:
            return texts.copy(), []

        corrected = texts.copy()
        corrections: List[Correction] = []

        for i, text in enumerate(texts):

            if confidences and confidences[i] >= self.confidence_threshold:
                continue

            words = text.split()
            corrected_words = []

            for j, word in enumerate(words):
                next_word = ""
                if j < len(words) - 1:

                    next_word = words[j + 1]

                corrected_word = self.correct_word(
                    word,
                    next_word
                )
        
                corrected_words.append(corrected_word)

                if corrected_word != word:
                    corrections.append(
                        Correction(
                        original_text=word,
                        corrected_text=corrected_word,
                        confidence=(
                                confidences[i]
                                if confidences
                                else 0.0
                            ),
                        position=i,
                        reason="dictionary+bigram"
                        )   
                    )

            corrected[i] = " ".join(corrected_words)

        return corrected, corrections
    
    
    
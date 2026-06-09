import re
from typing import List
from .correction_type import Correction


class RuleCorrector:
    
    def __init__(self):
        # Các rule phổ biến OCR tiếng Việt
        self.rules = [
            (re.compile(r'l(\d)'), r'1\1'),           # l1 → 11
            (re.compile(r'(\d)l'), r'\g<1>1'),           # 1l → 11
            (re.compile(r'O(\d)'), r'0\1'),           # O1 → 01
            (re.compile(r'(\d)O'), r'\g<1>0'),           # 10 → 100
            (re.compile(r'\s+'), ' '),                # nhiều khoảng trắng
            (re.compile(r' - '), '-'),                # khoảng trắng quanh dấu gạch ngang
        ]

    def correct(self, texts: List[str]) -> tuple[List[str], List[Correction]]:
        corrected = texts.copy()
        corrections = []
        
        for i, text in enumerate(texts):
            original = text
            for pattern, repl in self.rules:
                new_text = pattern.sub(repl, text)
                if new_text != text:
                    corrections.append(Correction(
                        original_text=original,
                        corrected_text=new_text,
                        confidence=0.0,      # rule-based nên confidence cao
                        position=i,
                        reason="rule-based"
                    ))
                    text = new_text
            corrected[i] = text
            
        return corrected, corrections
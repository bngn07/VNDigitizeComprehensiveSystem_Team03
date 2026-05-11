import jiwer
from dataclasses import dataclass


@dataclass
class EvaluationScore:
    cer: float
    wer: float
    
    def __str__(self):
        return f"CER: {self.cer * 100:.2f}% | WER: {self.wer * 100:.2f}%"

class OCREvaluator:
    @staticmethod
    def clean_text(text: str) -> str:
        cleaned = " ".join(text.split()).lower()
        return cleaned

    def evaluate_result(self, ground_truth: str, ocr_result) -> EvaluationScore:
        if isinstance(ocr_result.texts, str):
            predicted_text = ocr_result.texts
        else:
            predicted_text = " ".join([block.text for block in ocr_result.texts])
            
        clean_truth = self.clean_text(ground_truth)
        clean_pred = self.clean_text(predicted_text)
        
        if not clean_truth:
            return EvaluationScore(cer=1.0, wer=1.0)
            
        cer_score = jiwer.cer(clean_truth, clean_pred)
        wer_score = jiwer.wer(clean_truth, clean_pred)
        
        return EvaluationScore(cer=cer_score, wer=wer_score)
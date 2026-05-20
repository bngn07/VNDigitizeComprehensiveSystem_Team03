from .autocorrect import AutoCorrector
from .rule import RuleCorrector
from .dictionary import DictionaryCorrector
from .correction_type import Correction, CorrectionResult

# Export ra ngoài
__all__ = [
    'AutoCorrector',
    'RuleCorrector',
    'DictionaryCorrector',
    'Correction',
    'CorrectionResult'
]


print("Correction module loaded successfully")  
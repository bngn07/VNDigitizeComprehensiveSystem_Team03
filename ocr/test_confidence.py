# ====================================================================
#       test_confidence.py
#
#   Unit test cho ConfidenceScorer (Task 3)
#   Chạy: python ocr/test_confidence.py
# ====================================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ocr.schema import WordResult, OCRResult
from ocr.confidence import ConfidenceScorer

scorer = ConfidenceScorer(threshold=0.8)

# ====================================================================
# CASE 1: Mix từ rõ và mờ (case cơ bản)
# ====================================================================
case1 = OCRResult(
    words=[
        WordResult(text="Quyết",  confidence=0.95, x=10,  y=10, width=50, height=20),
        WordResult(text="định",   confidence=0.91, x=65,  y=10, width=40, height=20),
        WordResult(text="số",     confidence=0.55, x=110, y=10, width=20, height=20),
        WordResult(text="123/QĐ", confidence=0.42, x=135, y=10, width=60, height=20),
        WordResult(text="UBND",   confidence=0.88, x=200, y=10, width=50, height=20),
    ]
)
result1 = scorer.score(case1)
print("=== CASE 1: Mix từ rõ và mờ ===")
print(result1)
assert result1.words[0].flagged == False, "Quyết không được flag"
assert result1.words[1].flagged == False, "định không được flag"
assert result1.words[2].flagged == True,  "số phải bị flag"
assert result1.words[3].flagged == True,  "123/QĐ phải bị flag"
assert result1.words[4].flagged == False, "UBND không được flag"
assert len(scorer.get_flagged(result1)) == 2, "Phải có đúng 2 từ bị flag"
assert round(result1.overall_confidence, 3) == 0.742
print(" Pass\n")

# ====================================================================
# CASE 2: Tất cả từ đều rõ
# ====================================================================
case2 = OCRResult(
    words=[
        WordResult(text="Cộng", confidence=0.97, x=10,  y=10, width=50, height=20),
        WordResult(text="hòa",  confidence=0.95, x=65,  y=10, width=40, height=20),
        WordResult(text="xã",   confidence=0.93, x=110, y=10, width=30, height=20),
        WordResult(text="hội",  confidence=0.98, x=145, y=10, width=30, height=20),
    ]
)
result2 = scorer.score(case2)
print("=== CASE 2: Tất cả từ đều rõ ===")
print(result2)
assert len(scorer.get_flagged(result2)) == 0, "Không có từ nào bị flag"
print(" Pass\n")

# ====================================================================
# CASE 3: Tất cả từ đều mờ
# ====================================================================
case3 = OCRResult(
    words=[
        WordResult(text="mờ1", confidence=0.30, x=10,  y=10, width=50, height=20),
        WordResult(text="mờ2", confidence=0.45, x=65,  y=10, width=40, height=20),
        WordResult(text="mờ3", confidence=0.20, x=110, y=10, width=30, height=20),
    ]
)
result3 = scorer.score(case3)
print("=== CASE 3: Tất cả từ đều mờ ===")
print(result3)
assert len(scorer.get_flagged(result3)) == 3, "Tất cả phải bị flag"
print(" Pass\n")

# ====================================================================
# CASE 4: Không có từ nào (list rỗng)
# ====================================================================
case4 = OCRResult(words=[])
result4 = scorer.score(case4)
print("=== CASE 4: Không có từ ===")
print(result4)
assert result4.overall_confidence == 0.0
assert len(scorer.get_flagged(result4)) == 0
print(" Pass\n")

# ====================================================================
# CASE 5: Đúng ngưỡng (confidence = 0.8 chính xác)
# ====================================================================
case5 = OCRResult(
    words=[
        WordResult(text="đúng_ngưỡng", confidence=0.80, x=10, y=10, width=50, height=20),
        WordResult(text="dưới_ngưỡng", confidence=0.79, x=65, y=10, width=50, height=20),
    ]
)
result5 = scorer.score(case5)
print("=== CASE 5: Đúng ngưỡng 0.8 ===")
print(result5)
assert result5.words[0].flagged == False, "0.80 không được flag"
assert result5.words[1].flagged == True,  "0.79 phải bị flag"
print(" Pass\n")

# ====================================================================
# CASE 6: Thay đổi ngưỡng lên 95%
# ====================================================================
scorer_strict = ConfidenceScorer(threshold=0.95)
case6 = OCRResult(
    words=[
        WordResult(text="Quyết", confidence=0.95, x=10, y=10, width=50, height=20),
        WordResult(text="định",  confidence=0.91, x=65, y=10, width=40, height=20),
    ]
)
result6 = scorer_strict.score(case6)
print("=== CASE 6: Ngưỡng 95% ===")
print(result6)
assert result6.words[0].flagged == False, "0.95 không bị flag với ngưỡng 0.95"
assert result6.words[1].flagged == True,  "0.91 phải bị flag với ngưỡng 0.95"
print(" Pass\n")

# ====================================================================
# CASE 7: Ngưỡng thấp 50%
# ====================================================================
scorer_loose = ConfidenceScorer(threshold=0.5)
case7 = OCRResult(
    words=[
        WordResult(text="rõ",  confidence=0.90, x=10, y=10, width=50, height=20),
        WordResult(text="mờ",  confidence=0.55, x=65, y=10, width=40, height=20),
        WordResult(text="hỏng",confidence=0.40, x=110, y=10, width=40, height=20),
    ]
)
result7 = scorer_loose.score(case7)
print("=== CASE 7: Ngưỡng 50% ===")
print(result7)
assert result7.words[0].flagged == False, "0.90 không bị flag với ngưỡng 0.5"
assert result7.words[1].flagged == False, "0.55 không bị flag với ngưỡng 0.5"
assert result7.words[2].flagged == True,  "0.40 phải bị flag với ngưỡng 0.5"
print(" Pass\n")

# ====================================================================
# CASE 8: Ngưỡng không hợp lệ
# ====================================================================
print("=== CASE 8: Ngưỡng không hợp lệ ===")
try:
    scorer_invalid = ConfidenceScorer(threshold=1.5)
    print(" Fail — phải raise ValueError")
except ValueError as e:
    print(f"Bắt được lỗi đúng: {e}")
    print(" Pass\n")

# ====================================================================
print("=" * 40)
print(" Tất cả test đều pass!")
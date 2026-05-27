import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
from src.correction.autocorrect import AutoCorrector

if __name__ == "__main__":

    config = {
        "enabled": True,
        "min_confidence": 0.8
    }

    corrector = AutoCorrector(config)

    test_cases = [
        "So 0l/QD-UBND",
        "Ngay 2O/05/2O24",
        "Thoi gian 0l:3O",
        "Dia chi 12 l0 Nguyen Hue",
        "Can cu vao hop dong lao dong",
        "Uy ban nhan dan thanh pho",
        "Dia chi lien he",
        "Thoi gian thuc hien",
        "Cong hoa xa hoi chu nghia Viet Nam",
        "Ngay hop mat cuoi nam",
        "Dia diem hop lop",
        "Thoi tiet hom nay dep",
        "Can ban nha mat tien",
        "Nhan vien hanh chinh",
        "Hop d0ng lao d0ng",
        "Uy ban nhan d4n",
        "Thoi gi4n lam viec",
        "Dja chi lien he",
        "Thong bao tuyen dung",
        "Quyet dinh so 12/QD-UBND",
        "Can cu theo quy dinh hien hanh",
        "Ngay 0l thang 05 nam 2O24"
    ]

    confidences = [
        0.40,
        0.35,
        0.30,
        0.42,
        0.45,
        0.50,
        0.48,
        0.52,
        0.43,
        0.95,
        0.96,
        0.97,
        0.94,
        0.98,
        0.38,
        0.41,
        0.36,
        0.39,
        0.60,
        0.55,
        0.58,
        0.37
    ]

    result = corrector.correct_list(
        test_cases,
        confidences
    )

    print(
        f"\nTotal corrections: "
        f"{result.corrected_count}\n"
    )

    if result.corrections:

        for corr in result.corrections:

            print(
                f"✓ "
                f"{corr.original_text}"
                f" -> "
                f"{corr.corrected_text} "
                f"({corr.reason})"
            )

    else:
        print("Không có correction nào.")


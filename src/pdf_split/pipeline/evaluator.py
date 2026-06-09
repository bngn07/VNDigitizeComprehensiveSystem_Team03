"""
Đánh giá kết quả pipeline so với ground truth.
"""


def evaluate(result: dict, gt_types: dict[int, str]) -> None:
    """
    In bảng so sánh dự đoán vs ground truth và tính accuracy.

    Tham số:
        result   : payload trả về từ run_pipeline()
        gt_types : dict {page_start: label} — ground truth
    """
    if "error_code" in result:
        print(f"❌ {result['error_code']}: {result['error_message']}")
        return

    sub_docs = result.get("sub_documents", [])
    correct = total = 0

    print(f"{'Seg':>4} {'Trang':>12} {'Dự đoán':>22} {'Ground Truth':>22}  OK?")
    print("-" * 70)

    for i, doc in enumerate(sub_docs, start=1):
        gt = gt_types.get(doc["page_start"])
        if not gt:
            continue
        is_correct = doc["type"] == gt
        correct   += is_correct
        total     += 1
        print(
            f"{i:>4} "
            f"p{doc['page_start']}-{doc['page_end']:<3}  "
            f"{doc['type']:>22}  "
            f"{gt:>22}  "
            f"{'✅' if is_correct else '❌'}"
        )

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.1f}%")
    else:
        print("\n⚠️  Không có ground truth nào khớp với các segment.")

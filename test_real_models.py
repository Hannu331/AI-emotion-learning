"""
Verifies the real trained BiLSTM and BERT models actually load and
produce valid predictions (not crashing, not placeholder-random).

Run from the project root:  python test_real_models.py
"""

from src.predict import predict_emotion, predict_both
from src.keyword_rules import EMOTIONS

TEST_CASES = [
    ("I am so lost on recursion, nothing makes sense", "Confused"),
    ("I finally understand how loops work, this is easy now", "Confident"),
    ("This lecture is the same thing every week, so dull", "Bored"),
    ("I wonder how neural networks actually learn patterns", "Curious"),
    ("I've tried everything and I still can't get this to work, so annoying", "Frustrated"),
]


def check_scores_valid(scores, label):
    checks = []

    checks.append((f"{label}: all 5 emotions present",
                    set(scores.keys()) == set(EMOTIONS)))

    total = sum(scores.values())
    checks.append((f"{label}: scores sum to ~1.0 (got {total:.4f})",
                    abs(total - 1.0) < 0.01))

    checks.append((f"{label}: all scores between 0 and 1",
                    all(0.0 <= v <= 1.0 for v in scores.values())))

    return checks


def main():
    all_checks = []

    print("=== Testing BiLSTM predictions ===")
    for text, expected_hint in TEST_CASES:
        try:
            scores = predict_emotion(text, "bilstm")
            top = max(scores, key=scores.get)
            print(f"  '{text[:50]}...' -> top: {top} ({scores[top]:.2%})  [expected-ish: {expected_hint}]")
            all_checks.extend(check_scores_valid(scores, f"BiLSTM '{text[:20]}...'"))
        except Exception as e:
            print(f"  [FAIL] Exception on BiLSTM prediction: {e}")
            all_checks.append((f"BiLSTM prediction for '{text[:20]}...' ran without error", False))

    print("\n=== Testing BERT predictions ===")
    for text, expected_hint in TEST_CASES:
        try:
            scores = predict_emotion(text, "bert")
            top = max(scores, key=scores.get)
            print(f"  '{text[:50]}...' -> top: {top} ({scores[top]:.2%})  [expected-ish: {expected_hint}]")
            all_checks.extend(check_scores_valid(scores, f"BERT '{text[:20]}...'"))
        except Exception as e:
            print(f"  [FAIL] Exception on BERT prediction: {e}")
            all_checks.append((f"BERT prediction for '{text[:20]}...' ran without error", False))

    print("\n=== Testing predict_both (compare mode) ===")
    try:
        both = predict_both(TEST_CASES[0][0])
        all_checks.append(("predict_both returns both 'bilstm' and 'bert' keys",
                            set(both.keys()) == {"bilstm", "bert"}))
    except Exception as e:
        print(f"  [FAIL] Exception in predict_both: {e}")
        all_checks.append(("predict_both ran without error", False))

    print("\n=== RESULTS ===")
    all_passed = True
    for label, passed in all_checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")

    print("\nOVERALL:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED (see above)")
    print("\nNote: 'expected-ish' emotions are a rough sanity guide, not a")
    print("strict grading of model accuracy — real model accuracy depends")
    print("on your training data quality. This script mainly confirms the")
    print("models load and produce well-formed output without crashing.")


if __name__ == "__main__":
    main()

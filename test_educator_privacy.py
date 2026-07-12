"""
Automated test for the educator-linking / privacy feature.
Run from the project root:  python test_educator_privacy.py

This bypasses the Streamlit UI entirely and talks to the database
directly, so you get instant pass/fail instead of manual clicking.

NOTE: This inserts real test rows into your actual data/app.db.
Re-running it multiple times is safe (uses unique emails via a
random suffix) but you may want to clean up test rows afterward.
"""

import random
import string

from src.database import (
    init_db, create_user, verify_email, set_user_status,
    set_linked_educator, get_active_educators,
    insert_emotion_record, get_records_for_educator, get_records_for_user,
    get_connection,
)


def rand_suffix():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def force_verify(email):
    """Bypass real email for testing: mark verified directly in DB."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE Users SET email_verified = 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()


def main():
    init_db()
    suffix = rand_suffix()

    print("=== Setting up test educator ===")
    educator_email = f"test_educator_{suffix}@example.com"
    ok, msg = create_user(educator_email, "Test Educator", "TestPass123!", "educator")
    print("create educator:", ok, msg)
    force_verify(educator_email)
    set_user_status(educator_email, "active")  # approve, skipping admin panel

    print("\n=== Setting up Student A (linked to educator) ===")
    student_a_email = f"test_student_a_{suffix}@example.com"
    ok, msg = create_user(student_a_email, "Student A", "TestPass123!", "student")
    print("create student A:", ok, msg)
    force_verify(student_a_email)
    set_linked_educator(student_a_email, educator_email)

    print("\n=== Setting up Student B (private, no educator) ===")
    student_b_email = f"test_student_b_{suffix}@example.com"
    ok, msg = create_user(student_b_email, "Student B", "TestPass123!", "student")
    print("create student B:", ok, msg)
    force_verify(student_b_email)
    set_linked_educator(student_b_email, None)

    print("\n=== Logging one record for each student ===")
    insert_emotion_record(
        student_a_email, "Math", "I am lost on recursion", "Confused", "Frustrated",
        0.8, "bilstm", "Try breaking it down.", "ai", "{}", False,
    )
    insert_emotion_record(
        student_b_email, "Physics", "This is private, should not be seen", "Bored", None,
        0.7, "bilstm", "Stay engaged.", "ai", "{}", False,
    )

    print("\n=== Checking educator's visible records ===")
    educator_records = get_records_for_educator(educator_email)
    visible_emails = {r["email"] for r in educator_records}
    print("Educator sees records from:", visible_emails)

    checks = []
    checks.append(("Educator SEES Student A", student_a_email in visible_emails))
    checks.append(("Educator does NOT see Student B", student_b_email not in visible_emails))

    print("\n=== Checking each student sees only their own records ===")
    a_own = get_records_for_user(student_a_email)
    b_own = get_records_for_user(student_b_email)
    checks.append(("Student A sees their own record", len(a_own) == 1))
    checks.append(("Student B sees their own record", len(b_own) == 1))

    print("\n=== RESULTS ===")
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")

    print("\nOVERALL:", "ALL TESTS PASSED (check marks above)" if all_passed else "SOME TESTS FAILED (see above)")
    print(f"\n(Test accounts created with suffix '{suffix}' - safe to leave in DB, or delete manually if desired)")


if __name__ == "__main__":
    main()

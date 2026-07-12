"""
Automated test for export_user_data() and delete_user_account().
Run from the project root:  python test_export_delete.py

Creates one throwaway test account, logs a record for it, exports its
data, deletes the account, then verifies it's actually gone.
"""

import random
import string

from src.database import (
    init_db, create_user, get_connection,
    insert_emotion_record, export_user_data, delete_user_account,
    ADMIN_EMAIL,
)


def rand_suffix():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def force_verify(email):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE Users SET email_verified = 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()


def user_exists(email):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email FROM Users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return row is not None


def records_exist_for(email):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT record_id FROM Emotion_Records WHERE email = ?", (email,))
    rows = c.fetchall()
    conn.close()
    return len(rows) > 0


def main():
    init_db()
    suffix = rand_suffix()
    checks = []

    print("=== Creating test account ===")
    email = f"test_exportdel_{suffix}@example.com"
    ok, msg = create_user(email, "Export Delete Test", "TestPass123!", "student")
    print("create_user:", ok, msg)
    force_verify(email)

    print("\n=== Logging a test record ===")
    insert_emotion_record(
        email, "Chemistry", "test input for export/delete check", "Curious", None,
        0.6, "bert", "test ai response", "ai", "{}", False,
    )

    print("\n=== Testing export_user_data ===")
    exported = export_user_data(email)
    checks.append(("export_user_data returns a result", exported is not None))
    if exported:
        checks.append(("export includes 'profile' key", "profile" in exported))
        checks.append(("export includes 'emotion_records' key", "emotion_records" in exported))
        checks.append(("exported profile has no password field",
                        "password" not in exported.get("profile", {})))
        checks.append(("exported records include the one we logged",
                        len(exported.get("emotion_records", [])) == 1))

    print("\n=== Testing delete_user_account ===")
    ok, msg = delete_user_account(email)
    print("delete_user_account:", ok, msg)
    checks.append(("delete_user_account reports success", ok))
    checks.append(("user no longer exists in Users table", not user_exists(email)))
    checks.append(("user's records no longer exist", not records_exist_for(email)))

    print("\n=== Testing admin account is protected from deletion ===")
    ok, msg = delete_user_account(ADMIN_EMAIL)
    print("delete_user_account(admin):", ok, msg)
    checks.append(("admin account deletion is blocked", ok is False))
    checks.append(("admin account still exists", user_exists(ADMIN_EMAIL)))

    print("\n=== RESULTS ===")
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")

    print("\nOVERALL:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED (see above)")


if __name__ == "__main__":
    main()

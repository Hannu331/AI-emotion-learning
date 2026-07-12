"""
Deletes test accounts (and their emotion records) created by
test_educator_privacy.py. Safe to run anytime — only touches rows
where the email matches the test pattern used by that script.

Run from the project root:  python cleanup_test_data.py
"""

from src.database import get_connection


def main():
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT email FROM Users WHERE email LIKE 'test_%@example.com'")
    test_emails = [row["email"] for row in c.fetchall()]

    if not test_emails:
        print("No test accounts found. Nothing to clean up.")
        conn.close()
        return

    print(f"Found {len(test_emails)} test account(s):")
    for e in test_emails:
        print(" -", e)

    c.execute("DELETE FROM Emotion_Records WHERE email LIKE 'test_%@example.com'")
    records_deleted = c.rowcount
    c.execute("DELETE FROM Users WHERE email LIKE 'test_%@example.com'")
    users_deleted = c.rowcount

    conn.commit()
    conn.close()

    print(f"\nDeleted {users_deleted} test user(s) and {records_deleted} test record(s).")


if __name__ == "__main__":
    main()

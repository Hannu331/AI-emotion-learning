"""
Automated check: runs app.py headlessly (no browser) using Streamlit's
AppTest framework, logs in as admin, and verifies the emoji renders
correctly (not garbled) plus scans all source files for any remaining
mojibake patterns.

Run from the project root:  python test_emoji_rendering.py

Requires streamlit >= 1.28 (AppTest was added then). If this errors
with an import failure, run: pip install --upgrade streamlit
"""

import glob
import os
import sys


def check_source_files_for_mojibake():
    print("=== Scanning source files for known mojibake patterns ===")
    KNOWN_EMOJIS = ["🛡️", "📊", "💬", "🧠", "👤", "🔒", "⚠️", "🎓"]
    mojibake_map = {}
    for emoji in KNOWN_EMOJIS:
        mojibake = emoji.encode("utf-8").decode("cp1252", errors="ignore")
        if mojibake != emoji:
            mojibake_map[mojibake] = emoji

    files = ["app.py"] + glob.glob("src/*.py")
    all_clean = True
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        found = [g for g in mojibake_map if g in content]
        if found:
            all_clean = False
            print(f"  [FAIL] {path} still contains mojibake: {found}")
        else:
            print(f"  [PASS] {path} clean")
    return all_clean


def check_admin_panel_rendering():
    print("\n=== Running app.py headlessly with AppTest ===")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("  [SKIP] Your streamlit version doesn't support AppTest "
              "(needs >= 1.28). Run: pip install --upgrade streamlit")
        return None

    at = AppTest.from_file("app.py")
    at.run(timeout=15)

    if at.exception:
        print("  [FAIL] App raised an exception on startup:")
        for e in at.exception:
            print("   ", e)
        return False

    # Log in as admin via the actual login form fields
    try:
        at.text_input(key="login_email").set_value("admin@learningassistant.local")
        at.text_input(key="login_password").set_value("ChangeMe123!")

        login_button = None
        for b in at.button:
            if b.label == "Log In":
                login_button = b
                break
        if login_button:
            login_button.click()
            at.run(timeout=15)
        else:
            print("  [SKIP] Could not find a 'Log In' button. "
                  "Your button label may differ from 'Log In'.")
            return None
    except Exception as e:
        print(f"  [SKIP] Could not automate login (key names may differ in your app): {e}")
        print("  This usually means your text_input widgets don't use "
              "key='login_email' / key='login_password'. Check your "
              "login_screen() function's exact key= values and tell me "
              "if they're different, so I can adjust this script.")
        return None

    # Collect all rendered text (markdown, subheader, caption, etc.)
    rendered_texts = []
    for block in at.markdown:
        rendered_texts.append(block.value)
    for block in at.subheader:
        rendered_texts.append(block.value)
    for block in at.caption:
        rendered_texts.append(block.value)

    found_correct = any("🛡️" in t for t in rendered_texts)
    found_garbled = any("ðŸ›¡ï¸" in t for t in rendered_texts)

    if found_correct and not found_garbled:
        print("  [PASS] Admin panel renders 🛡️ correctly")
        return True
    elif found_garbled:
        print("  [FAIL] Admin panel still shows garbled emoji")
        return False
    else:
        print("  [INFO] Could not locate the admin panel heading text "
              "(may not have reached that screen). Rendered texts seen:")
        for t in rendered_texts[:10]:
            print("   -", t)
        return None


def main():
    files_clean = check_source_files_for_mojibake()
    render_result = check_admin_panel_rendering()

    print("\n=== SUMMARY ===")
    print("Source files clean of mojibake:", "YES" if files_clean else "NO")
    if render_result is True:
        print("Admin panel renders correctly: YES")
    elif render_result is False:
        print("Admin panel renders correctly: NO")
    else:
        print("Admin panel render check: inconclusive (see notes above) — "
              "confirm visually with 'streamlit run app.py' as a fallback.")


if __name__ == "__main__":
    main()

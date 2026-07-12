"""
Fixes garbled emoji text (mojibake) like "ðŸ›¡ï¸" that should read "🛡️".

This happens when UTF-8 bytes get misread as Windows-1252 (cp1252) and
then re-saved. Rather than trying to reverse the byte corruption
generically (which can lose data for some characters), this script
matches a known list of emojis used in this project and swaps their
garbled form back to the correct emoji directly.

Run from the project root:  python fix_encoding.py
It rewrites app.py and any src/*.py files in place. Always makes a
.bak backup first.
"""

import glob
import os
import shutil

TARGET_FILES = ["app.py"] + glob.glob("src/*.py")

# Emojis used anywhere in this project. Add more here if you spot
# other garbled symbols later.
KNOWN_EMOJIS = ["🛡️", "📊", "💬", "🧠", "👤", "🔒", "⚠️", "🎓"]


def build_mojibake_map():
    """Maps each known emoji's corrupted (mojibake) form back to the
    correct emoji."""
    mapping = {}
    for emoji in KNOWN_EMOJIS:
        mojibake = emoji.encode("utf-8").decode("cp1252", errors="ignore")
        if mojibake != emoji:
            mapping[mojibake] = emoji
    return mapping


def main():
    mojibake_map = build_mojibake_map()
    any_changes = False

    for path in TARGET_FILES:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        replacements_made = []

        for garbled, correct in mojibake_map.items():
            if garbled in content:
                count = content.count(garbled)
                content = content.replace(garbled, correct)
                replacements_made.append((garbled, correct, count))

        if content != original:
            any_changes = True
            print(f"\n--- Fixing {path} ---")
            for garbled, correct, count in replacements_made:
                print(f"  Replaced {count}x: {garbled!r} -> {correct}")

            backup_path = path + ".bak"
            shutil.copy(path, backup_path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved. Backup kept at: {backup_path}")
        else:
            print(f"No known mojibake found in {path} (already fine).")

    if not any_changes:
        print("\nNo files needed fixing.")
    else:
        print("\nDone. Re-run your app and check the emoji rendering.")
        print("If it all looks correct, you can delete the .bak files.")
        print("If you still see garbled symbols NOT in the list above,")
        print("paste an example and I'll add it to KNOWN_EMOJIS.")


if __name__ == "__main__":
    main()

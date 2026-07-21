"""
setup_tle_credentials.py — one-time local setup for Space-Track credentials.

Run this once, locally, after cloning the repo:

    python setup_tle_credentials.py

It creates `tle_credentials_local.py` in the same directory, which
`tle_updater.py` already tries to import first (see its top-of-file
try/except). That file is listed in .gitignore, so it never gets committed.

This script does NOT create a Space-Track account for you — you still
need to sign up yourself at https://www.space-track.org/auth/createAccount
(email verification + agreeing to their terms is part of that, by design;
not something a script should do on your behalf). This just saves you
from having to hand-write the credentials file afterward.

Nothing typed here is sent anywhere, logged, or printed back out.
"""

import getpass
import sys
from pathlib import Path

CRED_FILE = Path(__file__).parent / "tle_credentials_local.py"
GITIGNORE = Path(__file__).parent / ".gitignore"


def _check_gitignored() -> bool:
    if not GITIGNORE.exists():
        return False
    text = GITIGNORE.read_text()
    return "tle_credentials_local.py" in text


def main() -> int:
    if CRED_FILE.exists():
        print(f"{CRED_FILE.name} already exists — leaving it alone.")
        print("Delete it first if you want to re-enter credentials.")
        return 0

    if not _check_gitignored():
        print(
            f"WARNING: '{CRED_FILE.name}' is not listed in {GITIGNORE.name}.\n"
            "Add it before continuing, or this file could get committed by "
            "accident. Aborting — nothing written."
        )
        return 1

    print("Space-Track credential setup")
    print("(You need an existing account — sign up first at")
    print(" https://www.space-track.org/auth/createAccount if you haven't.)")
    print()

    email = input("Space-Track email: ").strip()
    if not email:
        print("No email entered — aborting, nothing written.")
        return 1

    password = getpass.getpass("Space-Track password (hidden as you type): ")
    if not password:
        print("No password entered — aborting, nothing written.")
        return 1

    content = (
        '"""Local-only Space-Track credentials — never committed (see .gitignore)."""\n'
        f"ST_USER     = {email!r}\n"
        f"ST_PASSWORD = {password!r}\n"
    )
    CRED_FILE.write_text(content)

    # Best-effort: restrict to the current user only, where supported.
    try:
        CRED_FILE.chmod(0o600)
    except (NotImplementedError, OSError):
        pass  # e.g. some Windows filesystems don't support POSIX chmod bits

    print(f"\nWrote {CRED_FILE.name}. tle_updater.py will pick it up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

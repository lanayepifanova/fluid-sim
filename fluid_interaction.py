#!/usr/bin/env python3
"""Backward-compatible entrypoint for the refactored app."""

from app import main


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()

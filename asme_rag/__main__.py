"""Entry point for `python -m asme_rag`. Forces UTF-8 output on Windows, then runs cli.main()."""
import sys

from asme_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())

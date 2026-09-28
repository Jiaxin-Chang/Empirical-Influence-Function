#!/usr/bin/env python3
"""Java valid predictions → semantic top-10 → own continue file → GPU 2 train.

Do not point this at port 8765. Start a viewer on 8772 first.

Example::

    python -m src.auto_java_sal_logicvalid_continue --max-tests 1 --skip-train
    python -m src.auto_java_sal_logicvalid_continue
"""

from __future__ import annotations

import sys

from src.auto_lang_sal_logicvalid_continue import SPECS, main

if __name__ == "__main__":
    sys.exit(main(SPECS["java"]))

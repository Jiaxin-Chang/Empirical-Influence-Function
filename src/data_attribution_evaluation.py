from __future__ import annotations

import sys

from src.attribution_evaluation import main


if __name__ == "__main__":
    main(["--skip-feature", *sys.argv[1:]])

#!/usr/bin/env python3
import sys

from task_pipeline import main

if __name__ == "__main__":
    sys.argv.insert(1, "validate_doc_raw")
    main()

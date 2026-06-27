#!/usr/bin/env python3
from task_pipeline import main
import sys

if __name__ == "__main__":
    sys.argv.insert(1, "merge_doc_raw")
    main()

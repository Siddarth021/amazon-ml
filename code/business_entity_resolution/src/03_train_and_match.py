import os
import sys

# Delegate to 03_train_and_val.py for unified 10-Fold CV & timestamped runs logging
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

import importlib
train_val_module = importlib.import_module("03_train_and_val")

if __name__ == "__main__":
    train_val_module.main()

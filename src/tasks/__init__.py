import glob
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

py_files = glob.glob(os.path.join(current_dir, "*.py"))

modules = [os.path.splitext(os.path.basename(file))[0] for file in py_files]

modules = [module for module in modules if module != "__init__"]

__all__ = modules


for module in modules:
    exec(f"from .{module} import *")

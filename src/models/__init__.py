import glob
import os

# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get a list of all Python files in the current directory
py_files = glob.glob(os.path.join(current_dir, "*.py"))

# Extract the module names from the file paths
modules = [os.path.splitext(os.path.basename(file))[0] for file in py_files]

# Exclude the __init__.py file from the list of modules
modules = [module for module in modules if module != "__init__"]

# Define the __all__ variable to expose the modules
__all__ = modules

# Automatically import all the modules
for module in modules:
    exec(f"from .{module} import *")

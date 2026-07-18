import os
import shutil

# 1. Define new directory structure
folders = ["src/models", "src/data", "src/utils"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "__init__.py"), "w") as f:
        pass # Create empty init files

# 2. Move existing fusion file
if os.path.exists("src/fusion.py"):
    shutil.move("src/fusion.py", "src/models/fusion.py")

print("Restructuring complete.")
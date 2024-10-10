import subprocess
import os

# Define the paths
qmd_file = "index.qmd"
nosite_script = "nosite.py"

# Step 1: Render the Quarto file
print(f"Rendering {qmd_file} with Quarto...")
subprocess.run(["quarto", "render", qmd_file])

# Step 2: Run the nosite.py script to move _site/ to docs/ and delete _site/
if os.path.exists(nosite_script):
    print(f"Running {nosite_script} to move _site to docs/ and delete _site...")
    subprocess.run(["python", nosite_script])
else:
    print(f"{nosite_script} not found.")

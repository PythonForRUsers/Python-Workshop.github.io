import subprocess
import os

# Define the paths
qmd_file = "index.qmd"
nosite_script = "nosite.py"


def find_session_qmd_files():
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # List to store the paths of found .qmd files
    session_qmd_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(current_dir):
        # Check if any folder in the current path contains 'session' (case-insensitive)
        if any("session" in dir_name.lower() for dir_name in root.split(os.sep)):
            # Look for .qmd files in the current folder containing 'session' in the name
            for file in files:
                if file.endswith(".qmd") and "session" in file.lower():
                    # Get the full path of the .qmd file
                    full_path = os.path.join(root, file)
                    session_qmd_files.append(full_path)

    return session_qmd_files


# Run the function and print the results
session_qmd_files = find_session_qmd_files()
for file_path in session_qmd_files:
    subprocess.run(["quarto", "render", file_path])

# Step 1: Render the Quarto file
print(f"Rendering {qmd_file} with Quarto...")
subprocess.run(["quarto", "render", qmd_file])


# Step 2: Run the nosite.py script to move _site/ to docs/ and delete _site/
if os.path.exists(nosite_script):
    print(f"Running {nosite_script} to move _site to docs/ and delete _site...")
    subprocess.run(["python", nosite_script])
else:
    print(f"{nosite_script} not found.")

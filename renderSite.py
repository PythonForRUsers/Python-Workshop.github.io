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

    # List directories in the current directory
    for item in os.listdir(current_dir):
        # print(item)
        item_path = os.path.join(current_dir, item)
        # print(item_path)
        # Check if it's a directory and if 'session' is in the directory name
        if os.path.isdir(item_path) and "session" in item.lower():
            # List the files in this directory
            for file in os.listdir(item_path):
                # Check if the file is a .qmd file and does not contain 'ignore' in the name
                if file.endswith(".qmd") and "ignore" not in file.lower():
                    full_path = os.path.join(item_path, file)
                    session_qmd_files.append(full_path)

    return session_qmd_files


# Run the function and print the results
session_qmd_files = find_session_qmd_files()

for file_path in session_qmd_files:
    print(file_path)
    subprocess.run(["quarto", "render", file_path])

# Step 1: Render the Quarto file (except those with "ignore" in the title)
files_to_render = ["index.qmd", "about.qmd", "link.qmd", "sessions.qmd"]

for qmd_file in files_to_render:
    if "ignore" not in qmd_file.lower():
        print(f"Rendering {qmd_file} with Quarto...")
        subprocess.run(["quarto", "render", qmd_file])


# Step 2: Run the nosite.py script to move _site/ to docs/ and delete _site/
if os.path.exists(nosite_script):
    print(f"Running {nosite_script} to move _site to docs/ and delete _site...")
    subprocess.run(["python", nosite_script])
else:
    print(f"{nosite_script} not found.")

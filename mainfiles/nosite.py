import shutil
import os

# Define source and destination directories
current_dir = os.path.dirname(os.path.realpath(__file__))

source_dir = os.path.join(current_dir, "_site")
destination_dir = "/docs"

# Copy contents of _site to docs/
if os.path.exists(source_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy all content from _site to docs
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(destination_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Remove the _site directory after copying
    shutil.rmtree(source_dir)
    print(f"Successfully copied _site to {destination_dir} and deleted _site.")
else:
    print(f"Source directory {source_dir} does not exist.")

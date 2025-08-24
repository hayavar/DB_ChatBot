import os

def remove_tempfiles():
# List of file paths to remove
    files_to_remove = [#put your foler path here
    ]

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

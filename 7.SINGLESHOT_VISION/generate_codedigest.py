import os

def create_code_digest(root_dir, output_file, ignore_folders=None):
    if ignore_folders is None:
        ignore_folders = {'.git', 'node_modules', '__pycache__', '.pytest_cache'}
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip unwanted folders
            dirnames[:] = [d for d in dirnames if d not in ignore_folders]
            
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(file_path, root_dir)
                
                # Optionally filter by file extensions
                if file_path.endswith(('.js', '.py', '.java', '.html', '.css', '.ts', '.jsx', '.json')):
                    outfile.write(f"\n\n--- FILE: {relative_path} ---\n\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}")

# Usage
create_code_digest('.', 'code_digest.txt')
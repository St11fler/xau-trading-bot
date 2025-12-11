import os
from pathlib import Path

def find_python_files(directory):
    """
    Recursively find all .py files in the given directory.

    :param directory: Path to the directory to search.
    :return: List of Path objects pointing to .py files.
    """
    path = Path(directory)
    python_files = list(path.rglob('*.py'))
    return python_files

def save_files_with_code(python_files, output_file):
    """
    Save filenames and their code to the output file, numbered sequentially.

    :param python_files: List of Path objects pointing to .py files.
    :param output_file: Path to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, file_path in enumerate(python_files, start=1):
            f_out.write(f"{idx}. Filename: {file_path}\n")
            f_out.write("```python\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    code = f_in.read()
                f_out.write(code)
            except Exception as e:
                f_out.write(f"# Error reading file: {e}\n")
            f_out.write("\n```\n\n")  # Markdown code block for better readability

def main():
    # Specify the directory to search. You can change this to any directory you want.
    target_directory = '.'  # Current directory
    # Specify the output file name
    output_filename = 'output.txt'

    print(f"Searching for Python files in '{target_directory}'...")
    python_files = find_python_files(target_directory)
    print(f"Found {len(python_files)} Python file(s).")

    print(f"Saving filenames and code to '{output_filename}'...")
    save_files_with_code(python_files, output_filename)
    print("Done!")

if __name__ == "__main__":
    main()

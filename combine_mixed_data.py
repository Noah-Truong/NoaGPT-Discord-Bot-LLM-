import os
import glob
from config import Config
def combine_text_files(input_folder, output_file_name="motherload_data.txt"):
    """
    Combines all .txt files from a specified folder into a single output file.

    Args:
        input_folder (str): The path to the folder containing the text files.
        output_file_name (str): The name of the file to save the combined content.
    """
    # Create the full path for the output file
    output_file_path = os.path.join(input_folder, output_file_name)

    # Use glob to find all .txt files in the specified folder
    # Sort the file paths to ensure a consistent order of concatenation
    file_paths = sorted(glob.glob(os.path.join(input_folder, "*.txt")))

    if not file_paths:
        print(f"No .txt files found in the folder: {input_folder}")
        return

    print(f"Combining {len(file_paths)} files into {output_file_name}...")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    # Add a newline character after each file's content for separation
                    outfile.write("\n")
                print(f"  - Added content from: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error reading file {os.path.basename(file_path)}: {e}")

    print(f"Merging completed. Combined content saved to: {output_file_path}")

combine_text_files(Config.MIXED_DATA_FOLDER)
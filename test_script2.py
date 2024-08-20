import os
import re

def get_most_recent_file(folder_path):
    # Regular expression pattern to extract the number from the filename
    pattern = re.compile(r'(\d+)_corners\.csv')

    # Initialize variables to keep track of the most recent file
    most_recent_file = None
    highest_number = -1

    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the number from the filename
            number = int(match.group(1))
            # Update the most recent file if this number is higher
            if number > highest_number:
                highest_number = number
                most_recent_file = filename

    if most_recent_file:
        # Return the full path to the most recent file
        return os.path.join(folder_path, most_recent_file)
    else:
        # If no file matches, return None or handle it as needed
        return None

# Example usage
folder_path = 'docker/competition_output'
most_recent_file_path = get_most_recent_file(folder_path)
if most_recent_file_path:
    print(f"The most recent file is: {most_recent_file_path}")
else:
    print("No files found.")

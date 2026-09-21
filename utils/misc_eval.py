import os
import ast

def add_entry(key, value, save_folder):
    file_path = str(os.path.join(save_folder, "log_stats.txt"))
    # Step 1: Read the existing dictionary from the file
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
            my_dict = ast.literal_eval(file_content)
    except FileNotFoundError:
        my_dict = {}

    # Step 2: Add new entries to the dictionary
    my_dict[key] = value

    # Step 3: Write the updated dictionary back to the file
    with open(file_path, "w") as file:
        file.write(str(my_dict))

def reset_log_stats(save_folder):
    file_path = str(os.path.join(save_folder, "log_stats.txt"))
    if os.path.exists(file_path):
        os.remove(file_path)
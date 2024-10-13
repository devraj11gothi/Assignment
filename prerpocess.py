import pandas as pd
import os
from tqdm import tqdm

# Specify the folder containing your test files
test_folder = r'C:\Users\ADMIN\Desktop\Codes\Infrrd\claude\dataset\dataset\val\boxes_transcripts'

# Specify the column names to be added
column_names = ['start_index', 'end_index', 'x_top_left', 'y_top_left', 'x_bottom_right', 'y_bottom_right', 'transcript', 'field']

# Process each file in the test folder
print("Adding column names to test files...")
for filename in tqdm(os.listdir(test_folder)):
    if filename.endswith('.tsv'):
        file_path = os.path.join(test_folder, filename)
        
        # Read the file without header
        df = pd.read_csv(file_path, sep=',', header=None)
        
        # Check if the number of columns match
        if len(df.columns) != len(column_names):
            print(f"Warning: {filename} has {len(df.columns)} columns, but {len(column_names)} column names provided.")
            print("Skipping this file.")
            continue
        
        # Assign the column names
        df.columns = column_names
        
        # Save the file with the same name, now including headers
        df.to_csv(file_path, sep=',', index=False)

print("Process complete. Column names have been added to all compatible TSV files in the test folder.")
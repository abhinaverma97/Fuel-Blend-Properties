import pandas as pd
import os

# Define file paths
single_output_file = os.path.join('outputs', 'single-output.csv')
multi_output_file = os.path.join('outputs', 'multi_output_submission_lgbm_multi.csv')
blended_output_file = os.path.join('outputs', 'blended_submission.csv')

# Check if input files exist
if not os.path.exists(single_output_file):
    print(f"Error: Input file not found at {single_output_file}")
    print("Please run single-output.py to generate it first.")
    exit()

if not os.path.exists(multi_output_file):
    print(f"Error: Input file not found at {multi_output_file}")
    exit()

# Load the two submission files
df_single = pd.read_csv(single_output_file)
df_multi = pd.read_csv(multi_output_file)

# Columns to be replaced
columns_to_replace = [
    'BlendProperty1',
    'BlendProperty5',
    'BlendProperty6',
    'BlendProperty7',
    'BlendProperty10',
]

# Ensure the columns exist in both dataframes before proceeding
missing_cols_single = [col for col in columns_to_replace if col not in df_single.columns]
missing_cols_multi = [col for col in columns_to_replace if col not in df_multi.columns]

if missing_cols_single:
    print(f"Error: Columns missing in single-output.csv: {missing_cols_single}")
    exit()
if missing_cols_multi:
    print(f"Error: Columns missing in multi_output_submission_lgbm_multi.csv: {missing_cols_multi}")
    exit()


# Create a copy of the single-output dataframe to modify
df_blended = df_single.copy()

# Replace the values for the specified columns
print(f"Replacing columns: {', '.join(columns_to_replace)}")
for col in columns_to_replace:
    df_blended[col] = df_multi[col]

# Save the new blended submission file
df_blended.to_csv(blended_output_file, index=False)

print(f"\nSuccessfully blended submissions.")
print(f"Replaced {len(columns_to_replace)} columns in '{os.path.basename(single_output_file)}' with values from '{os.path.basename(multi_output_file)}'.")
print(f"Result saved to '{os.path.basename(blended_output_file)}'.") 
import pandas as pd
import os
import sys

# Check if file exists
filename = 'final_edge3.csv'
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found in current directory.")
    print("Current directory:", os.getcwd())
    print("Files in current directory:")
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            print(f"  - {f}")
    sys.exit(1)

try:
    # Read the CSV file
    df = pd.read_csv(filename, sep="\t")
    print(f"Successfully loaded {filename}")
    print(f"Data shape: {df.shape}")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Sort by score in descending order (highest scores first)
df_sorted_desc = df.sort_values('score', ascending=False)

# Sort by score in ascending order (lowest scores first)
df_sorted_asc = df.sort_values('score', ascending=True)

# Display the top 10 highest scoring edges
print("Top 10 highest scoring edges:")
print(df_sorted_desc.head(10))

print("\n" + "="*50 + "\n")

# Display the top 10 lowest scoring edges
print("Top 10 lowest scoring edges:")
print(df_sorted_asc.head(10))

print("\n" + "="*50 + "\n")

# Save sorted results to new CSV files
df_sorted_desc.to_csv('edges_sorted_by_score_desc.csv', index=False)

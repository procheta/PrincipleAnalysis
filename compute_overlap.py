import json
import pandas as pd

first_csv = "final_edges/final_edge_principal9.csv"
second_csv = "final_edges/final_edge_principal1.csv"

first_df = pd.read_csv(first_csv,sep="\t")
second_df = pd.read_csv(second_csv,sep="\t")

# Select top 400 edges from each (assuming sorted by score already)
top_400_first = first_df.head(400)
top_400_second = second_df.head(400)

# Represent edges as (source, target) tuples
edges_first = set(top_400_first['edge'])
edges_second = set(top_400_second['edge'])

overlap = edges_first & edges_second
print(f"Number of overlapping edges in top 400 is : {len(overlap)}")
print("Overlapping edges:")
for edge in overlap:
    #print(edge)
    # Store in a csv file
    overlap_df = pd.DataFrame(overlap, columns=['edge'])
    overlap_df.to_csv('overlap_edges.csv', index=False)

import pandas as pd
import re

input_file = "input.txt"
output_file = "output.xlsx"

with open(input_file, "r") as f:
    content = f.read()

blocks = content.strip().split("\n\n")

data = []

for block in blocks:
    row = {}
    lines = block.strip().split("\n")
    
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            row[key.strip()] = value.strip()
    
    # Extract seed from Model path
    if "Model" in row:
        match = re.search(r"seed(\d+)", row["Model"])
        if match:
            row["Seed"] = int(match.group(1))
    
    data.append(row)

df = pd.DataFrame(data)

# Optional: reorder columns nicely
columns_order = [
    "Thing", "Network File", "Seed", "Threshold",
    "F-score", "Recall", "Precision", "AUC", "AUPR", "Model"
]
df = df.reindex(columns=columns_order)

df.to_excel(output_file, index=False)

print("Done.")
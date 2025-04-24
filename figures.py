import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open("blip_test_output.json") as f:
    data = json.load(f)

# Convert to DataFrame and select first 10 unique caption-prediction pairs
df = pd.DataFrame(data)
sample_df = df.drop_duplicates(subset=["caption", "prediction"]).tail(10)

# Create table-style visualization without image column
plt.figure(figsize=(12, 6))
plt.axis('off')
plt.title("📷 Actual Image Description vs. BLIP Image Captioning", fontsize=14, weight="bold")

# Format table data (without image column)
table_data = [
    ["Actual Caption", "BLIP Generated Caption", "BLEU Score"]
] + [
    [row["caption"], row["prediction"], f'{row["bleu"]:.4f}']
    for _, row in sample_df.iterrows()
]

# Create and display the table
table = plt.table(cellText=table_data, colLabels=None, cellLoc='left', loc='center',
                  cellColours=[["#f2f2f2"]*3] + [["#ffffff"]*3]*len(sample_df))
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

plt.tight_layout()
plt.savefig("blip_caption_comparison_table_tail.png")
plt.show()

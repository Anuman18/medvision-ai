import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to dataset CSV
csv_path = 'D:\medvision-ai\data\chest_xrays\Data_Entry_2017.csv'

# Load dataset
print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Show basic info
print("\n✅ Dataset Loaded")
print("Total records:", len(df))
print("\n📋 Sample records:")
print(df.head())
# Count the frequency of each condition (can be multiple in one row)
disease_counts = df['Finding Labels'].str.split('|').explode().value_counts()

print("\n🦠 Top 10 diseases in dataset:")
print(disease_counts.head(10))

# Plot disease frequency
plt.figure(figsize=(12, 6))
sns.barplot(x=disease_counts.index, y=disease_counts.values, palette="magma")
plt.xticks(rotation=45)
plt.title("Distribution of Diseases in Chest X-ray Dataset")
plt.xlabel("Disease")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

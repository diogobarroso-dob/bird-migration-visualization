import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Data
# Ensure the CSV file is in the same directory or provide the full path
df = pd.read_csv('data/bird_migration.csv')

# 2. Select Numerical Columns for Correlation
# We filter out non-numeric columns (like 'bird_name', 'date_time')
# and irrelevant IDs (like 'device_info_serial')
numerical_cols = ['altitude', 'speed_2d', 'latitude', 'longitude', 'direction']
correlation_df = df[numerical_cols]

# 3. Calculate the Correlation Matrix
corr_matrix = correlation_df.corr()

# Print the numerical matrix
print("Correlation Matrix:")
print(corr_matrix)

# 4. Visualize with a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,         # Show the numbers on the squares
    cmap='coolwarm',    # Blue-Red color scale
    vmin=-1, vmax=1,    # Set scale range from -1 to 1
    center=0,           # Center the color scale at 0
    fmt=".2f"           # Format numbers to 2 decimal places
)
plt.title("Feature Correlation Matrix")
plt.show()
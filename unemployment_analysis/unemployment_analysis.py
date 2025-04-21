import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the datasets
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Strip whitespace from column names BEFORE anything else
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Show basic structure before cleaning
df1_info = df1.info()
df2_info = df2.info()

# Display null counts
df1_nulls = df1.isnull().sum()
df2_nulls = df2.isnull().sum()

# Show duplicates if any
df1_duplicates = df1.duplicated().sum()
df2_duplicates = df2.duplicated().sum()

# Convert 'Date' column to datetime format
df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)

# Show cleaning results
print(df1_info)
print(df1_nulls)
print("Duplicates in df1:", df1_duplicates)

print(df2_info)
print(df2_nulls)
print("Duplicates in df2:", df2_duplicates)



#Explatory Data Analysis
# Set style
sns.set(style="whitegrid")

#Overall unemployment trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=df2, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate Over Time by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

#Regional Insights: Top regions by unemployment
##Average unemployment per region
top_regions = df2.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_regions.values, y=top_regions.index, palette='Reds')
plt.title('Top 10 Regions by Average Unemployment Rate')
plt.xlabel('Avg. Unemployment Rate (%)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()
plt.close()

#Temporal Comparison (Jan 2020 vs. Nov 2020)
# Filter for Jan and Nov 2020

# Ensure the 'Date' column is converted to datetime
df2['Date'] = pd.to_datetime(df2['Date'])

# Filter data for January and the available months (Feb to Oct)
months_to_compare = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Jan (1) to Oct (10)
df_filtered = df2[df2['Date'].dt.month.isin(months_to_compare)]

# Create a list to store the comparison results
monthly_comparisons = []

# Iterate over each month (Feb to Oct) and compare it to January
for month in months_to_compare[1:]:  # Start from month 2 (Feb) to 10 (Oct)
    # Filter data for January and the current month
    jan_data = df_filtered[df_filtered['Date'].dt.month == 1]
    month_data = df_filtered[df_filtered['Date'].dt.month == month]

    # Align the data based on Region
    jan_unemp = jan_data.set_index('Region')['Estimated Unemployment Rate (%)']
    month_unemp = month_data.set_index('Region')['Estimated Unemployment Rate (%)']

    # Strip and lowercase the Region index for alignment
    jan_unemp.index = jan_unemp.index.str.strip().str.lower()
    month_unemp.index = month_unemp.index.str.strip().str.lower()

    # Perform the comparison (subtraction)
    change = (month_unemp - jan_unemp).dropna()

    # Store the comparison result
    monthly_comparisons.append(change)

# Create a DataFrame to hold all the comparison results
comparison_df = pd.DataFrame(monthly_comparisons).T
comparison_df.columns = [f"Month {month}" for month in months_to_compare[1:]]  # Label the columns
comparison_df.index = jan_unemp.index  # Use region names as the index

# Plotting the comparison results as a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(comparison_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("📈 Monthly Unemployment Rate Change: January vs. Feb-Oct 2020")
plt.xlabel("Month")
plt.ylabel("Region")
plt.tight_layout()
plt.show()
plt.close()

# Output the results for insights
print("Top 5 Regions by Avg Unemployment Rate:")
print(top_regions.head())  # Display top 5 regions by average unemployment

# If you have a sorted comparison dataframe, print the largest increase and decrease
# Sort the final comparison dataframe
compare_sorted = comparison_df.mean(axis=1).sort_values()
print("\nLargest Increase from Jan to Nov 2020:")
print(compare_sorted.head())  # Top regions with the largest increase in unemployment

print("\nLargest Decrease from Jan to Nov 2020:")
print(compare_sorted.tail())  # Top regions with the largest decrease in unemployment

from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset only once and store in 'df'
df = pd.read_csv("Advertising.csv")

# Drop the 'Unnamed: 0' column (likely an index column)
df.drop(columns=["Unnamed: 0"], inplace=True)

# Check if the column was successfully dropped
df.head()
print(df.head())



# # Visualize correlations 
# sns.heatmap(date.corr(), annot=True)
# plt.title("Feature Correlation Heatmap")
# plt.show()

# # Exploratory Data Analysis (EDA)
# sns.pairplot(df, diag_kind='kde')
# plt.show()

# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Feature Correlation")
# plt.show()

#Feature Selection & Splitting
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Enhanced Actual vs Predicted Sales Plot
plt.figure(figsize=(12, 8))

# Create scatter plot with color gradient based on error magnitude
residuals = y_test - y_pred
abs_errors = np.abs(residuals)
scatter = plt.scatter(y_test, y_pred, c=abs_errors, cmap='viridis', 
                     alpha=0.7, edgecolors='w', s=100, label='Predictions')

# Add perfect prediction line (y=x)
perfect_line = np.linspace(min(y_test)-1, max(y_test)+1, 100)
plt.plot(perfect_line, perfect_line, 'r--', lw=2, label='Perfect Prediction')

# Add residual lines
for actual, pred in zip(y_test, y_pred):
    plt.plot([actual, actual], [actual, pred], 'gray', alpha=0.2)

# Add regression line for trend
sns.regplot(x=y_test, y=y_pred, scatter=False, 
            line_kws={'color':'orange', 'lw':2, 'alpha':0.5})

# Add metrics and information
plt.text(0.05, 0.95, 
         f'R² = {r2_score(y_test, y_pred):.3f}\n'
         f'MSE = {mean_squared_error(y_test, y_pred):.3f}\n'
         f'Avg Error = {np.mean(abs_errors):.2f} units\n'
         f'n = {len(y_test)} samples',
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8),
         fontsize=10)

# Add colorbar for error magnitude
cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (units)', rotation=270, labelpad=15)

# Customize the plot
plt.xlabel('Actual Sales (thousands of units)', fontsize=12)
plt.ylabel('Predicted Sales (thousands of units)', fontsize=12)
plt.title('Enhanced Actual vs Predicted Sales Analysis', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Add marginal histograms using seaborn (alternative to jointplot)
sns.histplot(y_test, color='blue', label='Actual', kde=True, alpha=0.3)
sns.histplot(y_pred, color='orange', label='Predicted', kde=True, alpha=0.3)
plt.legend()
plt.show()
#Saving the Model
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as sales_model.pkl")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Load data
df = pd.read_csv(r"C:\Users\U S E R\Desktop\Data Science\OASIS INFOBYTE\TASKS\car_price_prediction\data\car data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Explotory Data Analysis
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

df.dropna(inplace=True)
df['brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])


# df = df.drop(columns=['Car_Name'])  

df.drop(columns=['Car_Name'], inplace=True, errors='ignore')




# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


#train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R2 Score:", r2_score(y_test, y_pred))



# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_preds = rf_model.predict(X_test)

print("Random Forest MAE:", mean_absolute_error(y_test, rf_preds))
print("Random Forest MSE:", mean_squared_error(y_test, rf_preds))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("Random Forest R2 Score:", r2_score(y_test, rf_preds))

#feature importance visualisation
importances = rf_model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.show()

# Scatter plot: Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()






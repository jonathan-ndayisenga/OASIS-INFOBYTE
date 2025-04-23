import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib


# Load dataset
df = pd.read_csv("iris.csv")

# Explore the data
print(df.head())
print(df["Species"].value_counts())

# Preprocess
X = df.drop(['Id', 'Species'], axis=1)  # Features
y = df['Species']    

le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier().fit(X, y)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# # Evaluate
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Visualize
# sns.pairplot(df, hue="Species")
# plt.suptitle("Iris Flower Pairplot", y=1.02)
# plt.show()


# # After model training (e.g., model = LogisticRegression())
# y_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)

# # Plot with labels
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix for Iris Classification')
# plt.savefig('confusion_matrix.png')  # Save for Streamlit
# plt.show()


joblib.dump(model, 'iris_model_fixed.pkl') # Save for Streamlit
print(type(model))  
model = joblib.load('iris_model.pkl')
print("Model features:", model.feature_names_in_)  # If available


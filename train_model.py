import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Car.csv")

# Check first 5 rows
print(df.head())

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# One-Hot Encoding
X = pd.get_dummies(X)

# Save encoded column names
encoded_columns = X.columns

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical columns
numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and objects
joblib.dump(model, "LR_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoded_columns, "columns.pkl")

print("Model saved successfully!")
print("LR_model.pkl created")
print("scaler.pkl created")
print("columns.pkl created")
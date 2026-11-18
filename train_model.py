import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris classification dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train a Decision Tree classifier
model = DecisionTreeClassifier(
    criterion="gini", max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))

# Save the trained model and class labels
joblib.dump(
    {"model": model, "target_names": iris.target_names.tolist()},
    "iris_model.joblib"
)
print("Model saved as iris_model.joblib")

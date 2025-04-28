import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import our Decision Tree implementation
from DT import DecisionTreeClassifier

# Load a dataset (iris dataset)
data = datasets.load_iris()
print("Dataset loaded: Iris dataset")
print(f"Features: {data.feature_names}")
print(f"Target classes: {data.target_names}")

# Create a DataFrame
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target
print("\nDataFrame shape:", df.shape)
print(df.head())

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train the Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(
    criteria="gini",
    max_depth=5,
    sample_split_min=2,
    sample_leaf_min=1
)
dt_classifier.fit(X_train, y_train)

# Make predictions
print("Making predictions on test set...")
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:")
print(conf_matrix)

# Export the tree structure (if available)
print("\nDecision Tree Structure:")
dt_classifier._export_tree()

print("\nTest completed successfully!")
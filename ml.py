# Naive Bayes on Iris Dataset with Confusion Matrix Heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
# 2. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 3. Train Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Prediction
y_pred = model.predict(X_test)

# 5. Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# 7. Heatmap Visualization
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap (Naive Bayes - Iris)")
plt.show()
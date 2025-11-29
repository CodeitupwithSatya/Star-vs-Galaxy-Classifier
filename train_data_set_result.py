import numpy as np
import pandas as pd

# 1. Load and clean data
df = pd.read_csv('PS1_star_galaxy_classification_train.csv')
df = df.drop_duplicates()
df = df.fillna(df.median())

# 2. Train/test split (80%/20%), shuffled
from numpy.random import default_rng
rng = default_rng(seed=42)
indices = rng.permutation(len(df))
split_point = int(0.8 * len(df))
train_idx, test_idx = indices[:split_point], indices[split_point:]
train = df.iloc[train_idx]
test = df.iloc[test_idx]

X_train = train.drop('label', axis=1).values
y_train = train['label'].values.reshape(-1, 1)
X_test = test.drop('label', axis=1).values
y_test = test['label'].values.reshape(-1, 1)

# 3. Standard scaling (fitted on train, applied to both)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std

# 4. Manual logistic regression training (gradient descent)
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))
def fit_logreg(X, y, lr=0.05, iters=1500):
    m, n = X.shape
    w = np.zeros((n, 1)); b = 0.0
    for _ in range(iters):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        dz = y_pred - y
        dw = np.dot(X.T, dz) / m
        db = np.sum(dz) / m
        w -= lr * dw
        b -= lr * db
    return w, b
def predict_logreg(X, w, b):
    z = np.dot(X, w) + b
    p = sigmoid(z)
    output = (p >= 0.5).astype(int)
    return output, p

w, b = fit_logreg(X_train_s, y_train, lr=0.01, iters=1500)
train_pred, train_probs = predict_logreg(X_train_s, w, b)
test_pred, test_probs = predict_logreg(X_test_s, w, b)

train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)
print(f"The learnt learning rate and bias:{w,b}")
print(f"Training Accuracy: {train_acc*100:.2f}")
print(f"Test Accuracy: {test_acc*100:.2f}")

# 5. Confusion matrix for test set
def confusion_matrix(y_true, y_pred):
    y_true = y_true.flatten(); y_pred = y_pred.flatten()
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])
cm = confusion_matrix(y_test, test_pred)
print("Confusion Matrix (Test):\n", cm)

# 6. Save predictions with probabilities for test set
result_df = test.copy()
result_df['Predicted_Label'] = test_pred.flatten()
result_df['Probability_Galaxy'] = test_probs.flatten()
result_df.to_csv('train_set_predictions.csv', index=False)

print("Predictions for test set saved to test_set_predictions.csv.")
print("First 5 test predictions:")
print(result_df[['Predicted_Label', 'Probability_Galaxy']].head().to_string(index=False))

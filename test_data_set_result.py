import numpy as np
import pandas as pd

from train_data_set_result import mean,std,w,b

# Load the test data
test_df = pd.read_excel('test_galaxy_star.xlsx')

# Preprocess test data
X_test_raw = test_df.values  #all columns in test_df are features (no label)
# Scale using training statistics
X_test_scaled = (X_test_raw - mean) / std

# Predict using logistic regression model parameters
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.dot(X_test_scaled, w) + b  # Linear combination
probs = sigmoid(z)                # Predicted probabilities of class 1 (galaxy)
pred_labels = (probs >= 0.5).astype(int)  # Threshold at 0.5 for binary class prediction

# Prepare results dataframe to save predictions
results_df = test_df.copy()
results_df['Predicted_Label'] = pred_labels.flatten()
results_df['Probability_Galaxy'] = probs.flatten()
results_df['Probability_Star'] = (1 - probs).flatten()


# Save to CSV
results_df.to_csv('test_set_predictions.csv', index=False)

print("Predictions saved to 'test_set_predictions.csv'")
print("First 5 prediction rows:")
print(results_df[['Predicted_Label', 'Probability_Galaxy']].head())
num_stars = (results_df['Predicted_Label'] == 0).sum()
num_galaxies = (results_df['Predicted_Label'] == 1).sum()
print(f"Total stars predicted: {num_stars}")
print(f"Total galaxies predicted: {num_galaxies}")

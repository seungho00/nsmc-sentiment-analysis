import dataset
import pandas as pd

(_, y_train), (_, y_valid), (_, y_test) = dataset.load_data()

# Majority class
majority_label = y_train.value_counts().idxmax()
'''
print(y_train.value_counts(normalize=True))

0    0.501152
1    0.498848
Name: proportion, dtype: float64
'''

accuracy_valid = (y_valid == majority_label).mean()
print(f"Validation accuracy: {accuracy_valid:.2%}")  # 50.11%

accuracy_test = (y_test == majority_label).mean()
print(f"Test accuracy: {accuracy_test:.2%}")  # 49.65%
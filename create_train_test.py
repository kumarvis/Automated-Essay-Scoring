import pandas as pd
from sklearn.model_selection import train_test_split

# Load the train.csv dataset into a DataFrame
df = pd.read_csv('dataset/train.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['score'])
y = df['score']

# Split the dataset into train and test sets (80/20 ratio) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Concatenate features and target for train and test sets
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save the test set to a separate file (e.g., test.csv)
test_df.to_csv('dataset/test_split.csv', index=False)

# Save the train set to a separate file (e.g., train_split.csv)
train_df.to_csv('dataset/train_split.csv', index=False)

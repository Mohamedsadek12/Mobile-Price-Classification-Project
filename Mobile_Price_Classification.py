import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 1- Check for missing values ----> no missing values
#print("Missing values in train set:\n", train_df.isnull().sum())
#print("\nMissing values in test set:\n", test_df.isnull().sum())

# 2- Remove duplicates if any
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# 3- Visualize correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(train_df.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Training Data")
plt.tight_layout()
plt.show()

# 4- Distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='price_range')
plt.title("Distribution of Price Range")
plt.xlabel("Price Range")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Pairplot of selected features
selected_features = ['battery_power', 'ram', 'px_height', 'px_width', 'price_range']
sns.pairplot(train_df[selected_features], hue='price_range')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# 6- Feature Selection

X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]

cor_matrix = X.corr().abs()

# Take the upper triangle only of the correlation matrix cus The correlation matrix is symmetric
upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Drop highly correlated features (>0.85)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
print("\nHighly Correlated Features Dropped:", to_drop)

X_filtered = X.drop(columns=to_drop)











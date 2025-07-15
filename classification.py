import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load user data and model
df = pd.read_csv("ad_users.csv")
with open("ad_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create features
df['combined'] = df['interests'] + ' ' + df['gender'] + ' ' + df['location'] + ' ' + df['age'].astype(str)
X_text = df['combined']

# Predict probabilities
prob_dict = model.predict_proba(X_text)

# Convert to DataFrame
prob_df = pd.DataFrame(prob_dict)

# Visualize heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(prob_df.iloc[:20], cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Top 20 Users - Classification Probabilities per Category")
plt.xlabel("Ad Category")
plt.ylabel("User Index")
plt.tight_layout()
plt.show()


# Choose a single ad category (e.g., 'technology')
category = 'technology'
probs = prob_df[category]
ages = df['age']

plt.figure(figsize=(8, 6))
sns.scatterplot(x=ages, y=probs)
sns.regplot(x=ages, y=probs, scatter=False, color='red')
plt.title(f"Ad Interest Probability vs. Age for '{category.title()}'")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.grid(True)
plt.show()


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Reduce to 2D using PCA
features = model.vectorizer.transform(X_text)
pca = PCA(n_components=2)
reduced = pca.fit_transform(features.toarray())

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(reduced)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=clusters, palette="Set2")
plt.title("User Clustering based on Interests (PCA + KMeans)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

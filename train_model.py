import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from model import UnifiedAdModel

# Load data
df = pd.read_csv("ad_users.csv")

# Define ad categories
ad_categories = [
    'cricket', 'bollywood', 'cooking', 'yoga', 'technology',
    'fashion', 'travel', 'spirituality', 'education', 'gaming',
    'music', 'finance', 'health', 'politics', 'startups'
]

# Create target labels
y = pd.DataFrame()
for category in ad_categories:
    y[category] = df['interests'].apply(lambda x: int(category in x.split()))

# Create combined features
df['combined_features'] = (
    df['interests'] + ' ' +
    df['gender'] + ' ' +
    df['location'] + ' ' +
    df['age'].astype(str)
)

# Vectorize features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['combined_features'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
models = {}
y_pred_all = []

print("\nTraining individual models...\n")
for category in ad_categories:
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train[category])
    models[category] = model

    # Predict on test set
    y_pred_all.append(model.predict(X_test))

# Combine predictions
y_pred_all = pd.DataFrame(y_pred_all).T  # Shape: [n_samples, n_categories]
y_pred_all.columns = ad_categories

# Evaluate overall performance
print("\nOverall Model Evaluation on Test Set:\n")

acc = accuracy_score(y_test, y_pred_all)
prec = precision_score(y_test, y_pred_all, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred_all, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_all, average='macro', zero_division=0)

print(f"  Accuracy (subset): {acc:.4f}")
print(f"  Precision (macro): {prec:.4f}")
print(f"  Recall (macro):    {rec:.4f}")
print(f"  F1 Score (macro):  {f1:.4f}")


unified_model = UnifiedAdModel(models, vectorizer)

with open("ad_model.pkl", "wb") as f:
    pickle.dump(unified_model, f)

with open("ad_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel training and evaluation complete. Artifacts saved:")
print("- ad_model.pkl")
print("- ad_vectorizer.pkl")

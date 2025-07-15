# AdRecommendationSystem
A machine learning-based Ad Recommendation System designed to deliver personalized advertisement categories based on user profiles. Built using classification models and clustering techniques, this system improves ad targeting, user engagement, and digital experience.

## 📌 Project Overview
In the era of digital marketing, generic ad targeting results in poor user engagement. Our system leverages user demographic and interest data to recommend ads that are most likely to be clicked, using supervised machine learning and unsupervised clustering techniques.

## 🎯 Problem Statement
Traditional ad systems rely on basic data like demographics or browsing history. This project aims to overcome such limitations by:
Analyzing deeper features: interests, age, gender, location.
Using machine learning classifiers to personalize ad suggestions.
Enhancing user experience and advertiser ROI with smart recommendations.

## 📊 Dataset Description

Format: CSV
Size: ~1000 records
Source: Public dataset
Target Variable: click (Binary: 0 = No click, 1 = Click)

## 📌 Key Features

Feature	Description
age	Age of the user
gender	Gender of the user
location	City/Country of the user
interests	Space-separated keywords indicating interests
click	Whether the ad was clicked (0 or 1)

## 🛠️ Methodology

1. 🔄 Data Preprocessing
Removed null or irrelevant columns (e.g., user_id)
Label encoding and one-hot encoding for categorical features
Normalized numerical features using Min-Max Scaling
Handled class imbalance using SMOTE (if needed)

2. 🤖 Model Building
Model Used: Random Forest Classifier
Techniques:
GridSearchCV for hyperparameter tuning
Multi-label classification for assigning ads across multiple categories
Input: Combined text vector of interests, gender, location, and age using TF-IDF

3. 🧪 Evaluation
Metrics: Accuracy, Precision, Recall, F1-score
Tools: Confusion Matrix, Feature Importance Plot

4. 📈 Clustering Analysis
Applied PCA + K-Means for user segmentation into 3 clusters:
Cluster 1: Tech/Gaming
Cluster 2: Fashion/Lifestyle
Cluster 3: Education/Finance

## 🖥️ GUI
Built using Tkinter for local use.

Features:
User input form for age, gender, location, interests
Display of recommended ad categories
Clean, user-friendly layout

## 📁 Project Structure

├── ad_users.csv                # Input dataset
├── model.py                    # Unified model code
├── ad_recommendation.py       # Training & prediction script
├── gui_tkinter.py              # Tkinter-based interface
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

## 💡 Future Improvements
✅ Add advanced models like XGBoost and Deep Neural Networks
✅ Deploy real-time system via Flask or Streamlit
✅ Include Clickstream behavior for enhanced targeting
✅ Automate feedback loop to improve model accuracy

## Installation & Usage

# Clone the repository
git clone https://github.com/your-username/ad-recommendation-system.git
cd ad-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the model training
python ad_recommendation.py

# Launch the GUI
python gui_tkinter.py

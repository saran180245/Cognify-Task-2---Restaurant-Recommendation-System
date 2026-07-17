import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 50)
print("Restaurant Recommendation System")
print("=" * 50)

# Load Dataset
df = pd.read_csv("Dataset.csv", encoding="latin1")

print("\n✅ Dataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())
# ===============================
# Data Preprocessing
# ===============================

print("\nChecking Missing Values...\n")

print(df.isnull().sum())
# ===============================
# Handle Missing Values
# ===============================

df["Cuisines"] = df["Cuisines"].fillna("Unknown")

print("\n✅ Missing Values After Cleaning:\n")
print(df.isnull().sum())
# ===============================
# Remove Duplicate Restaurants
# ===============================

before = len(df)

df = df.drop_duplicates(subset="Restaurant Name")

after = len(df)

print("\nDuplicate Removal")
print("Rows Before :", before)
print("Rows After  :", after)
# ===============================
# Select Required Columns
# ===============================

df = df[
    [
        "Restaurant Name",
        "City",
        "Cuisines",
        "Average Cost for two",
        "Price range",
        "Aggregate rating",
        "Votes"
    ]
]

print("\nSelected Columns:")
print(df.head())
# ===============================
# Feature Engineering
# ===============================

df["Features"] = (
    df["Cuisines"].astype(str) + " " +
    df["City"].astype(str) + " " +
    df["Price range"].astype(str)
)

print("\nFeature Column Created Successfully!\n")

print(df[["Restaurant Name", "Features"]].head())
# ============================================
# TF-IDF Vectorization
# ============================================

vectorizer = TfidfVectorizer(stop_words="english")

feature_matrix = vectorizer.fit_transform(df["Features"])

print("\n✅ TF-IDF Matrix Created Successfully!")
print("Shape :", feature_matrix.shape)


# ============================================
# Cosine Similarity
# ============================================

similarity = cosine_similarity(feature_matrix)

print("\n✅ Cosine Similarity Matrix Created!")
print("Matrix Shape :", similarity.shape)
# ============================================
# Restaurant Recommendation Function
# ============================================

def recommend_restaurant(name):
    if name not in df["Restaurant Name"].values:
        print("\n❌ Restaurant Not Found!")
        return

    index = df[df["Restaurant Name"] == name].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\n🍽 Recommended Restaurants:\n")

    count = 0

    for i in scores[1:]:
        restaurant = df.iloc[i[0]]

        print("----------------------------------------")
        print("Restaurant :", restaurant["Restaurant Name"])
        print("City       :", restaurant["City"])
        print("Cuisine    :", restaurant["Cuisines"])
        print("Rating     :", restaurant["Aggregate rating"])
        print("----------------------------------------")

        count += 1

        if count == 5:
            break


# ============================================
# User Input
# ============================================

restaurant_name = input("\nEnter Restaurant Name: ")

recommend_restaurant(restaurant_name)
# 🍽️ Restaurant Recommendation System

A Machine Learning-based Restaurant Recommendation System that recommends similar restaurants based on **Cuisine**, **City**, and **Price Range** using **TF-IDF Vectorization** and **Cosine Similarity**.

##  Project Overview

This project uses a **Content-Based Recommendation System** to suggest restaurants that are similar to a user-selected restaurant.

The recommendation is based on restaurant features instead of user ratings or user history.

## 🚀 Features

- Load restaurant dataset
- Handle missing values
- Remove duplicate records
- Feature Engineering
- TF-IDF Vectorization
- Cosine Similarity Calculation
- Recommend Top 5 Similar Restaurants

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- VS Code


## 📂 Dataset

The dataset contains:

- **9551 Restaurant Records**
- **21 Columns**

Important columns used:

- Restaurant Name
- City
- Cuisines
- Average Cost for Two
- Price Range
- Aggregate Rating
- Votes


## ⚙️ Project Workflow

Load Dataset
      ↓
Data Preprocessing
      ↓
Handle Missing Values
      ↓
Remove Duplicates
      ↓
Feature Engineering
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Restaurant Recommendation
```

---

## 📊 Data Preprocessing

- Filled missing values in the **Cuisines** column.
- Removed duplicate restaurant names.
- Selected only the required columns for recommendation.
- Created a new **Features** column by combining:
  - Cuisine
  - City
  - Price Range

---

## 🤖 Machine Learning Technique

### Content-Based Recommendation System

The project recommends restaurants based on similar features.

### Algorithms Used

- TF-IDF Vectorizer
- Cosine Similarity

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Restaurant-Recommendation-System.git
```

Move into the project folder

```bash
cd Restaurant-Recommendation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## 💻 Sample Output

```
Enter Restaurant Name:

Le Petit Souffle

Recommended Restaurants

• Izakaya Kikufuji
• Tokyo Mon Amour
• Bonne Bouche
• Le Cirque - The Leela Palace
• Chez Jerome - Q Cafe


## 🎯 Future Improvements

- Filter recommendations by minimum rating
- Budget-based recommendations
- Location-based recommendations
- Web application using Flask or Streamlit
- Interactive User Interface



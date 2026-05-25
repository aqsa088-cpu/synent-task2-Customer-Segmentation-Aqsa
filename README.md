# Customer Segmentation with K-Means Clustering

> Unsupervised machine learning to identify distinct customer groups from mall shopping behavior data.

---

## Problem Statement

Retail businesses often treat all customers alike, missing opportunities to tailor marketing, offers, and experiences to specific audiences. This project applies unsupervised machine learning to automatically discover natural groupings within a customer base — enabling targeted marketing strategies, personalized recommendations, and smarter resource allocation without any labeled training data.

The core question: **Can we group mall customers into meaningful segments based on their demographics and spending behavior?**

---

## Dataset Details

**Source:** Mall Customers dataset (`Mall_Customers.csv`)

| Feature | Type | Description |
|---|---|---|
| `CustomerID` | Integer | Unique customer identifier |
| `Genre` | Categorical | Customer gender (Male / Female) |
| `Age` | Integer | Customer age in years |
| `Annual Income (k$)` | Integer | Annual income in thousands of USD |
| `Spending Score (1-100)` | Integer | Mall-assigned score based on purchasing behavior and frequency |

**Size:** 200 customer records, no missing values.

**Preprocessing:**
- One-hot encoding applied to the `Genre` column (`Genre_Male` binary feature, `drop_first=True` to avoid multicollinearity)
- All features standardized using `StandardScaler` prior to clustering (zero mean, unit variance)

---

## Approach

### 1. Exploratory Data Analysis
- Loaded and inspected the dataset for structure, types, and missing values
- Confirmed data cleanliness — no imputation required

### 2. Feature Engineering
- Encoded the categorical `Genre` column into a binary numeric feature
- Selected four clustering features: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`, `Genre_Male`

### 3. Feature Scaling
- Applied `StandardScaler` to normalize features to comparable scales, preventing high-magnitude features (e.g. income) from dominating distance calculations

### 4. Optimal K Selection — Elbow Method
- Ran K-Means for K = 1 to 10, recording Within-Cluster Sum of Squares (WCSS) for each
- Plotted WCSS vs. K to identify the "elbow" — the point of diminishing returns in cluster compactness
- **Selected K = 5** as the optimal number of clusters based on the elbow curve

### 5. K-Means Clustering
- Algorithm: `KMeans` with `k-means++` initialization, `random_state=42`, `n_init=10`
- Fit on scaled features; cluster labels appended to the original DataFrame

### 6. Visualization & Interpretation
- Scatter plots of key feature pairs colored by cluster:
  - **Annual Income vs. Spending Score** — primary segmentation view
  - **Age vs. Spending Score** — demographic-behavioral view
- Computed per-cluster mean values for all features to characterize each segment

---

## Results

K-Means identified **5 distinct customer segments**. Based on cluster mean analysis, the segments broadly correspond to the following archetypes:

| Cluster | Income | Spending Score | Age | Profile |
|---|---|---|---|---|
| 0 | Medium | Medium | Middle-aged | Average customers — moderate in all dimensions |
| 1 | High | Low | Older | High earners, conservative spenders — likely value-focused |
| 2 | Low | High | Young | Budget-constrained but high-engagement shoppers |
| 3 | High | High | Young–Middle | Prime targets — affluent and actively spending |
| 4 | Low | Low | Older | Low engagement, price-sensitive segment |

> *Exact cluster numbers and profiles may vary by run; interpret via the cluster characteristic means table.*

### Key Takeaways
- **Income vs. Spending Score** was the most discriminating feature pair, producing the clearest visual separation between clusters
- The **high-income / high-spending** segment (Cluster 3) represents the highest-value marketing target
- The **low-income / high-spending** segment (Cluster 2) may signal loyalty program opportunities
- Age played a secondary but meaningful role in differentiating behavioral patterns within income bands

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Preprocessing (`StandardScaler`) and clustering (`KMeans`) |
| `matplotlib` | Elbow method plot |
| `seaborn` | Cluster scatter plots |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 2. Place Mall_Customers.csv in /content/ (or update the path in the notebook)

# 3. Open and run the notebook
jupyter notebook Customer_Segmentation.ipynb
```

---

## Project Structure

```
.
├── Customer_Segmentation.ipynb   # Main analysis notebook
├── Mall_Customers.csv            # Input dataset
└── README.md                     # This file
```

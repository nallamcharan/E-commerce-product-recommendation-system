import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load dataset
df = pd.read_csv("amazon_30k_products.csv")

# Handle missing values
df['description'] = df['description'].fillna("")
df['offers'] = df['offers'].fillna(df['offers'].mode()[0])
df['product_name'] = df['product_name'].fillna("")
df = df.reset_index(drop=True)

# Create TF-IDF matrix for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])


# Content-based recommendation function
def recommend_product(product_name, top_n=5):
    matches = df[df["product_name"].str.lower() == product_name.lower()]
    
    if matches.empty:
        return "Product not found"

    index = matches.index[0]

    # Compute cosine similarity for selected product
    similarity_scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    # Get top similar products excluding itself
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]

    return df[['product_name','brand','category',
               'price','rating','offers']].iloc[similar_indices]


# Create simulated user-product interaction matrix
n_users = 3000
n_products = len(df)
user_product_matrix = np.random.randint(0, 2, (n_users, n_products))

# Apply matrix factorization using SVD
svd = TruncatedSVD(n_components=50, random_state=42)
user_embeddings = svd.fit_transform(user_product_matrix)
item_embeddings = svd.components_.T


# Collaborative filtering recommendation function
def collaborative_recommend(user_id, top_n=5):
    if user_id >= n_users or user_id < 0:
        return "Invalid User ID"

    user_vector = user_embeddings[user_id]

    # Predict product scores
    scores = np.dot(item_embeddings, user_vector)

    top_products = np.argsort(scores)[-top_n:][::-1]

    return df[['product_name','brand','category',
               'price','rating','offers']].iloc[top_products]


# Hybrid recommendation function
def hybrid_recommend(user_id, product_name, top_n=5):
    matches = df[df["product_name"].str.lower() == product_name.lower()]
    
    if matches.empty:
        return "Product not found"

    if user_id >= n_users or user_id < 0:
        return "Invalid User ID"

    index = matches.index[0]

    # Content score
    content_scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    # Collaborative score
    user_vector = user_embeddings[user_id]
    collab_scores = np.dot(item_embeddings, user_vector)

    # Normalize both scores
    scaler = MinMaxScaler()
    content_scores = scaler.fit_transform(
        content_scores.reshape(-1,1)
    ).flatten()

    collab_scores = scaler.fit_transform(
        collab_scores.reshape(-1,1)
    ).flatten()

    # Combine scores
    hybrid_scores = 0.6 * content_scores + 0.4 * collab_scores

    top_products = np.argsort(hybrid_scores)[-top_n:][::-1]

    return df[['product_name','brand','category',
               'price','rating','offers']].iloc[top_products]


# Testing
if __name__ == "__main__":
    print("\nContent Based Recommendation")
    print(recommend_product(df['product_name'].iloc[10]))

    print("\nCollaborative Recommendation")
    print(collaborative_recommend(10))

    print("\nHybrid Recommendation")
    print(hybrid_recommend(10, df['product_name'].iloc[10]))

model_data = {
    "tfidf": tfidf,
    "tfidf_matrix": tfidf_matrix,
    "user_embeddings": user_embeddings,
    "item_embeddings": item_embeddings,
    "dataframe": df,
    "n_users": n_users
}

with open("recommendation_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

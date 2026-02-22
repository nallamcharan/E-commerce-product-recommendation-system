from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
app = Flask(__name__)

# Load model
with open("recommendation_model.pkl", "rb") as f:
    model_data = pickle.load(f)
tfidf = model_data['tfidf']
tfidf_matrix =  model_data['tfidf_matrix']
user_embeddings = model_data['user_embeddings']
Df   = model_data['dataframe']
item_embeddings = model_data['item_embeddings']
n_users = model_data['n_users']

@app.route("/")
def home():
    return render_template('home.html')

#Content based recommendation system
@app.route("/",methods = ["POST",'GET'])

def recommend( top_n=5):
    product_name = request.form['productname']
    matches = Df[Df["product_name"].str.lower() == product_name.lower()]
    
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
    
    
    result  = Df[["product_name",'brand','category','rating','price']].iloc[similar_indices]

    return render_template('home.html',data = result)


if __name__ == "__main__":
    app.run(debug=True)
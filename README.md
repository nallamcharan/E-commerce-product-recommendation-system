🛍 Amazon Product Recommendation System
📌 Problem Statement

In large-scale e-commerce platforms like Amazon, customers face difficulty in discovering relevant products due to the vast number of available items. This often leads to:

Low user engagement

Reduced conversion rates

Poor customer experience

Decreased repeat purchases

The goal of this project is to build an intelligent Product Recommendation System that suggests relevant products to users using multiple recommendation techniques.

🎯 Project Objective

To design and implement three types of recommendation systems:

Content-Based Filtering

Collaborative Filtering

Hybrid Recommendation System

The system recommends products based on:

Product descriptions

User-product interactions

Combined scoring approach

📊 Dataset Overview

30,000 Amazon products

Features include:

Product Name

Brand

Category

Price

Rating

Rating Count

Stock Quantity

Delivery Days

Seller

Offers

Prime Availability

Return Policy

🧠 Approach
1️⃣ Content-Based Recommendation

Used TF-IDF Vectorization on product descriptions

Computed Cosine Similarity

Recommended products with highest similarity scores

📌 Advantage:

Does not require user history

Works for new users

2️⃣ Collaborative Filtering

Simulated user-product interaction matrix

Applied Matrix Factorization using Truncated SVD

Generated user and item embeddings

Predicted product scores using dot product

📌 Advantage:

Learns hidden patterns from user behavior

Personalized recommendations

3️⃣ Hybrid Recommendation System

Combined Content-Based and Collaborative scores

Applied MinMax Scaling

Final Score Formula:

Hybrid Score = 0.6 × Content Score + 0.4 × Collaborative Score

📌 Advantage:

Reduces cold start problem

Improves recommendation accuracy

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

TF-IDF Vectorizer

Cosine Similarity

TruncatedSVD

Pickle (Model Serialization)

Flask (For Deployment)

📁 Project Structure

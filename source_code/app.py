import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
import os
base = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base, 'dataset', 'products.csv') if os.path.exists(os.path.join(base, 'dataset')) else os.path.join('dataset', 'products.csv')
# fallback for both ways
try:
    df = pd.read_csv('dataset/products.csv')
except:
    df = pd.read_csv(csv_path)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])

def recommend_product(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = similarity.argmax()
    return df.iloc[best_match_index]

# Sample test
if __name__ == "__main__":
    query = "mera face oily hai"
    result = recommend_product(query)
    print(f"Input: {query}")
    print(f"Recommended: {result['product']} - {result['reason']}")
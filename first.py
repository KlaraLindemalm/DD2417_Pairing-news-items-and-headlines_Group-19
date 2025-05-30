# Read inout
import pandas as pd
import re

# Cosine and NN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


import numpy as np

FILE_PATH = 'news-article-categories.xlsx'

def split_on_comma_outside_quotes(s):
    return re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', s)

def read_file(filename):
    count = 0
    parsed_data = []
    titles = []
    articles = []
    df = pd.read_excel(filename)

    for id, combined_str in df['category,title,body'].items():
        if not isinstance(combined_str, str):
        # skip if not string
            count += 1
            continue
        
        parts = split_on_comma_outside_quotes(combined_str)
        if len(parts) == 3:
            category = parts[0].strip()
            title = parts[1].strip().strip('"').replace('""', '"')
            body = parts[2].strip().strip('"').replace('""', '"') 
            
            parsed_data.append({
                'category': category,
                'title': title,
                'body': body
            })
            titles.append({'id': id, 
                           'title':title})
            articles.append({'id': id, 
                             'article':body})
        else:
            count += 1
                
    return parsed_data, titles, articles, count

def nearest_neighbors(titles, articles):
    # Extract text and IDs
    title_texts = [t["title"] for t in titles]
    article_texts = [a["article"] for a in articles]
    title_ids = [t["id"] for t in titles]
    article_ids = [a["id"] for a in articles]

    # Vectorize all texts
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = title_texts + article_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split vectors back
    title_vecs = tfidf_matrix[:len(title_texts)]
    article_vecs = tfidf_matrix[len(title_texts):]

    # Fit NearestNeighbors on article vectors with cosine metric
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(article_vecs)

    # Find nearest article for each title
    distances, indices = nn.kneighbors(title_vecs)
    predicted_article_indices = indices.flatten()
    predicted_article_ids = [article_ids[i] for i in predicted_article_indices]

    # Evaluate accuracy
    correct_matches = sum(tid == aid for tid, aid in zip(title_ids, predicted_article_ids))
    accuracy = correct_matches / len(titles)
    print(f"Nearest Neighbors Accuracy: ", accuracy) # Term Frequency-Inverse Document Frequency

def main():
    parsed_data, titles, articles, count = read_file(FILE_PATH)
    nearest_neighbors(titles, articles)


if __name__ == "__main__":
    main()




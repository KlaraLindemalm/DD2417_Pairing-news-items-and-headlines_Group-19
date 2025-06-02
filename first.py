# Read inout
import pandas as pd
import re

# NN Cosine sim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# W2v Cosine sim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# bert cosine
from sentence_transformers import SentenceTransformer

# If not already downloaded
nltk.download('punkt')

# -------- Some variables ---------
FILE_PATH = 'news-article-categories.xlsx'
W2V_PATH = 'GoogleNews-vectors-negative300.bin'

# -------- Read the news data file and parse it into a useable format --------
def split_on_comma_outside_quotes(s):
    return re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', s) # This regular expression was created by CHAT GPT

def read_file(filename):
    titles = []
    articles = []
    df = pd.read_excel(filename)

    for _, combined_str in df['category,title,body'].items():
        if not isinstance(combined_str, str):
        # skip if not string some are NULL
            continue
        
        parts = split_on_comma_outside_quotes(combined_str)
        if len(parts) == 3:
            titles.append(parts[1].strip("\""))
            articles.append(parts[2].strip("\""))
                
    return titles, articles

# -------- Function to calculate and print accuracy --------
def accuracy(method, predicted_ids):
    correct = sum(i == pred for i, pred in enumerate(predicted_ids))
    accuracy = correct / len(predicted_ids)
    print(method, accuracy) 


# -------- Functions for NN cosine sim --------
def nearest_neighbors(titles, articles):

    # Vectorize all texts
    vectorizer = TfidfVectorizer(stop_words='english') # Term Frequency-Inverse Document Frequency
    all_texts = titles + articles
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split vectors back
    title_vecs = tfidf_matrix[:len(titles)]
    article_vecs = tfidf_matrix[len(articles):]

    # Fit and predict
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(article_vecs)
    _, indices = nn.kneighbors(title_vecs)
    predicted_ids = indices.flatten()

    # Evaluate accuracy
    accuracy("Nearest Neighbors Accuracy: ", predicted_ids)
    
# -------- functions for w2v cosine sim --------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    # Remove punctuation but keeps number isalpha was slightly better but i dont really understand why since title 
    # is likey to mention same numbers that the article does 
    return [token for token in tokens if token.isalnum()]  

def get_average_vector(tokens, model, dim=300):
    valid_tokens = [token for token in tokens if token in model]
    if not valid_tokens:
        return np.zeros(dim)    # returns a zero vector to avoid errors, this does happend a bit which is stange
    
    vectors = np.array([model[token] for token in valid_tokens])
    return np.mean(vectors, axis=0)

def word2vec_similarity(titles, articles):
    w2v_model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    title_vecs = []
    for title in titles:
        tokens = preprocess(title)
        vec = get_average_vector(tokens, w2v_model)
        title_vecs.append(vec)
    
    article_vecs = []
    for article in articles:
        tokens = preprocess(article)
        vec = get_average_vector(tokens, w2v_model)
        article_vecs.append(vec)
    
    # Convert to arrays
    title_vecs = np.array(title_vecs)
    article_vecs = np.array(article_vecs)

    # cosine similarity between each title and all articles
    sim_matrix = cosine_similarity(title_vecs, article_vecs)
    predicted_ids = sim_matrix.argmax(axis=1)

    # Evaluate 
    accuracy("Word2Vec Cosine: ", predicted_ids)

# ------ Bert scentence vectors with cosine sim ------
def bert_similarity(titles, articles):
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode titles and articles into embeddings
    title_vecs = bert_model.encode(titles, convert_to_tensor=False)
    article_vecs = bert_model.encode(articles, convert_to_tensor=False)

    # cosine similarity between each title and all articles
    sim_matrix = cosine_similarity(title_vecs, article_vecs)
    predicted_ids = sim_matrix.argmax(axis=1)

    accuracy("Bert cosine: ", predicted_ids)

# ------ Main() used to get the accuracy of the diffrent methods ------
def main():
    titles, articles = read_file(FILE_PATH)

    nearest_neighbors(titles, articles)

    word2vec_similarity(titles, articles)

    bert_similarity(titles, articles)

if __name__ == "__main__":
    main()




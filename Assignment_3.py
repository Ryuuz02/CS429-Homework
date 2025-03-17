import nltk
from nltk.corpus import state_union
from collections import Counter
import math

# Download NLTK resources if not already downloaded
nltk.download('state_union')

# Load State Union sample dataset
documents = state_union.fileids()

# Calculate TF-IDF scores
def calculate_tf_idf(documents):
    tf_idf_scores = {}
    document_frequencies = Counter()

    total_documents = len(documents)

    for document_id, document in enumerate(documents):
        # Tokenize the document
        tokens = nltk.word_tokenize(state_union.raw(document))
        term_frequencies = Counter(tokens)

        max_frequency = max(term_frequencies.values())

        for term, frequency in term_frequencies.items():
            # Calculate TF (Term Frequency) for each term in the document
            tf = 0.5 + 0.5 * (frequency / max_frequency)
            
            # Update document frequency for the term
            document_frequencies[term] += 1
            
            # Calculate IDF (Inverse Document Frequency) for each term
            idf = math.log10(total_documents / (1 + document_frequencies[term]))
            
            # Calculate TF-IDF score
            tf_idf = tf * idf
            
            # Update the inverted index
            if term not in tf_idf_scores:
                tf_idf_scores[term] = []
            tf_idf_scores[term].append((document, tf_idf))

    return tf_idf_scores

# Calculate TF-IDF scores
inverted_index = calculate_tf_idf(documents)

# Print example postings for a term
# example_term = 'the'
# if example_term in inverted_index:
#     print(f"Postings for term '{example_term}':")
#     for document, score in inverted_index[example_term]:
#         print(f"Document: {document}, TF-IDF Score: {score}")
# else:
#     print(f"No postings found for term '{example_term}'")

def convert_query_to_vector(query_terms, inverted_index):
    query_vector = {}
    
    # Calculate IDF scores for the query terms
    for term in query_terms:
        if term in inverted_index:
            # Retrieve IDF score for the term from the inverted index
            idf_score = inverted_index[term][0][1]  # Assuming IDF score is the same for all postings
            
            # Add the term and its IDF score to the query vector
            query_vector[term] = idf_score
    
    return query_vector

# Example usage:
query_terms = ['freedom', 'speech', 'rights']
query_vector = convert_query_to_vector(query_terms, inverted_index)
# print("Query Vector:")
# print(query_vector)

import numpy as np

def search(query_vector, inverted_index):
    document_scores = {}
    
    # Calculate document scores using cosine similarity
    for term, idf_score in query_vector.items():
        if term in inverted_index:
            for document, tf_idf_score in inverted_index[term]:
                if document not in document_scores:
                    document_scores[document] = 0
                document_scores[document] += tf_idf_score * idf_score
    
    # Sort document scores in descending order
    sorted_document_scores = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_document_scores

# Example usage:
sorted_results = search(query_vector, inverted_index)
print("Search Results:")
for document, score in sorted_results:
    print(f"Document: {document}, Score: {score}")


import nltk
nltk.download('state_union')
from nltk.corpus import state_union
from sklearn.datasets import fetch_20newsgroups
datalist = fetch_20newsgroups(subset='all', shuffle=False)
import pickle
import datetime

# Create a two-gram index for a list of sentences
def create_two_gram_index(sents):
    kgram_index = {}
    for sent in sents:
        for word in sent:
            for i in range(len(word) - 1):
                kgram = word[i:i+2]
                if kgram not in kgram_index:
                    kgram_index[kgram] = []
                kgram_index[kgram].append(word)
    return kgram_index

# Create an inverted index for a list of documents
def create_inverted_index(documents):
    inverted_index = {}
    for i in range(len(documents)):
        doc_sents = state_union.sents(fileids=documents[i])
        for sent in doc_sents:
            for word in sent:
                if word in inverted_index:
                    if i not in inverted_index[word]:
                        inverted_index[word].append(i)
                else:
                    inverted_index[word] = [i]
    return inverted_index

# Wildcard search
def wildcard_search(query, kgram_index, inverted_index):
    kgrams = [query[i:i+2] for i in range(len(query) - 1)]
    terms = set()
    for kgram in kgrams:
        if kgram in kgram_index:
            terms.update(kgram_index[kgram])
    filtered_terms = [term for term in terms if term.startswith(query)]
    document_ids = []
    for term in filtered_terms:
        document_ids.extend(inverted_index[term])
    return list(set(document_ids))

# Single term search
def single_term_search(query, kgram_index, inverted_index):
    if query in inverted_index:
        return inverted_index[query]
    else:
        suggestions = []
        for term in kgram_index.keys():
            distance = nltk.edit_distance(query, term)
            if distance <= 1:
                suggestions.append((term, distance))
        if suggestions:
            suggestions.sort(key=lambda x: x[1])  # Sort suggestions by edit distance
            suggested_term = suggestions[0][0]
            return f"Term not found. Did you mean '{suggested_term}'?"
        else:
            return "Term not found."

# Create a inverted index for one document in fetch_20newsgroups
def create_index_file(datalist):
    inverted_index = {}
    for i in range(len(datalist.data)):
        doc = datalist.data[i]
        doc_words = doc.split()
        for word in doc_words:
            if word in inverted_index:
                if i not in inverted_index[word]:
                    inverted_index[word].append(i)
            else:
                inverted_index[word] = [i]

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = f'inverted_index_{timestamp}.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump(inverted_index, f)

# Load multiple index files into a single inverted index
def load_index_files(file_names):
    inverted_index = {}
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            index = pickle.load(f)
            for word in index:
                if word in inverted_index:
                    inverted_index[word].extend(index[word])
                else:
                    inverted_index[word] = index[word]
    return inverted_index




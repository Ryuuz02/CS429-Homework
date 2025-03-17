import sklearn as sk
from sklearn import datasets
datalist = datasets.fetch_20newsgroups(subset='all', shuffle=False)
# Tokenize each document by removing white spaces and punctuation
def tokenizer(entry):
    entry = entry.lower()
    entry = entry.replace('\n', ' ')
    entry = entry.replace('\t', ' ')
    entry = entry.replace('\r', ' ')
    entry = entry.replace('!', ' ')
    entry = entry.replace('"', ' ')
    entry = entry.replace('#', ' ')
    entry = entry.replace('$', ' ')
    entry = entry.replace('%', ' ')
    entry = entry.replace('&', ' ')
    entry = entry.replace("'", ' ')
    entry = entry.replace('(', ' ')
    entry = entry.replace(')', ' ')
    entry = entry.replace('*', ' ')
    entry = entry.replace('+', ' ')
    entry = entry.replace(',', ' ')
    entry = entry.replace('-', ' ')
    entry = entry.replace('.', ' ')
    entry = entry.replace('/', ' ')
    entry = entry.replace(':', ' ')
    entry = entry.replace(';', ' ')
    entry = entry.replace('<', ' ')
    entry = entry.replace('=', ' ')
    entry = entry.replace('>', ' ')
    entry = entry.replace('?', ' ')
    entry = entry.replace('@', ' ')
    entry = entry.replace('[', ' ')
    entry = entry.replace('\\', ' ')
    entry = entry.replace(']', ' ')
    entry = entry.replace('^', ' ')
    entry = entry.replace('_', ' ')
    entry = entry.replace('`', ' ')
    entry = entry.replace('{', ' ')
    entry = entry.replace('|', ' ')
    entry = entry.replace('}', ' ')
    entry = entry.replace('~', ' ')
    entry = entry.replace('  ', ' ')
    entry = entry.split(' ')
    removal_counter = 0
    for word in entry:
        if word == '':
            removal_counter += 1
    for i in range(removal_counter):
        entry.remove('')
    return entry
tokenized_entries = []
for i in range(len(datalist.data)):
    tokenized_entries.append(tokenizer(datalist.data[i]))
# print(datalist.data[0])
# print(tokenized_entries[0])

# Make an inverted index of the tokenized entries
inverted_index = {}
for i in range(len(tokenized_entries)):
    for word in tokenized_entries[i]:
        if word in inverted_index:
            if i not in inverted_index[word]:
                inverted_index[word].append(i)
        else:
            inverted_index[word] = [i]

# Function for finding the intersection of two lists that progresses both lists one at a time to check if the word is in both lists
def intersection(lst1, lst2):
    i = 0
    j = 0
    lst3 = []
    while i < len(lst1) and j < len(lst2):
        if lst1[i] == lst2[j]:
            lst3.append(lst1[i])
            i += 1
            j += 1
        elif lst1[i] < lst2[j]:
            i += 1
        else:
            j += 1
    return lst3

# Function to sort list of words by frequency in the inverted index
def sort_by_frequency(word_lst):
    num_lst = []
    unsorted_lst = []
    sorted_lst = []
    for word in word_lst:
        frequency = len(inverted_index[word])
        num_lst.append(frequency)
        unsorted_lst.append(word)
    for i in range(len(num_lst)):
        min_index = num_lst.index(min(num_lst))
        sorted_lst.append(unsorted_lst[min_index])
        num_lst.pop(min_index)
        unsorted_lst.pop(min_index)
    return sorted_lst

# Unsure if the homework wants Large to Small or Small to Large since it doesn't say, this is just Small to Large

# Function to perform a search query given a string of words
def search_query(query):
    query = tokenizer(query)
    query = sort_by_frequency(query)
    result = inverted_index[query[0]]
    for i in range(1, len(query)):
        result = intersection(inverted_index[query[i]], result)
    return result
import nltk
import sys
import os
from nltk.tokenize import wordpunct_tokenize
import regex as re
import math

stopwords = nltk.corpus.stopwords.words("english")

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = dict()

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:            
            corpus[filename] = f.read()
    return corpus

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    list = []

    for item in wordpunct_tokenize(document):
        itemlower = item.lower()
        if re.search('[a-zA-Z]', itemlower) and itemlower not in stopwords :
            list.append(itemlower)

    return list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    IDF = dict()
    TD = len(documents.keys())

    for doc in documents:
        for word in documents[doc]:
            count = 0
            for doc2 in documents:
                if word in documents[doc2]:
                    count += 1
            if word not in IDF.keys():
                IDF[word] = math.log(TD/count)

    return IDF


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # for every word in query
        # For every File in files
            #find the term frequency of each query term in each file
            # compute TF-IDF
            # compute sum of all terms for each file
    
    # rank all files in Descending order of total TF-IDf values 

    TFIDF = dict()
 
    for word in query:
        for file in files:
            if file not in list(TFIDF.keys()):
                TFIDF[file] = 0
            if word not in idfs:
                continue
            count = files[file].count(word)
            TFIDF[file] += count*idfs[word]
    

    ranked = []   

    for i in range(n):
        v = -math.inf
        for file in TFIDF:
            if TFIDF[file] > v and file not in ranked:
                max = file
                v = TFIDF[file]
        ranked.append(max) 

    return ranked

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    senIDF = dict()

    for sen in sentences:
        for word in query:
            if sen not in list(senIDF.keys()):
                senIDF[sen] = 0
            if word not in idfs:
                continue
            if word in sentences[sen]:
                senIDF[sen] += idfs[word]

    ranked = []   

    for i in range(n):
        v = -math.inf
        for sen in senIDF:
            if senIDF[sen] > v and sen not in ranked:
                max = sen
                v = senIDF[sen]
            if senIDF[sen] == v and sen not in ranked:
                newSen = 0 
                oldSen = 0
                for word in query:
                    if word in sentences[sen]:
                        newSen += 1
                    if word in sentences[max]:
                        oldSen += 1
                if newSen/len(sentences[sen]) > oldSen/len(sentences[max]):
                    max = sen
                    v = senIDF[sen]

        ranked.append(max) 

    return ranked


if __name__ == "__main__":
    main()

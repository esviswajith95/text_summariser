import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re
from transformers import pipeline

# Read article
def read_article(text):

    cleaned_text = re.sub("\[(.*?)\]|\\n", " ", text)
    sentences = sent_tokenize(cleaned_text)
    
    return sentences

# create vectors form the sentences and calculate cosine similarity between these vectors
def sentence_similarity(sent1, sent2, stopwords=None):
    
    if  stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))
    vec1 = [0]*(len(all_words))
    vec2 = [0]*(len(all_words))

    #build vector for first sentence
    for w in sent1:
        if not w in stopwords:
            vec1[all_words.index(w)] += 1
    
    #build vector for second sentence
    for w in sent2:
        if not w in stopwords:
            vec2[all_words.index(w)] += 1
        
    return 1 - cosine_distance(vec1, vec2)

    # build similarity matrix

def build_similarity_matrix(sentences, stop_words):

    similarity_matrix = np.zeros((len(sentences),len(sentences)))

    for x1 in range(len(sentences)):
        for x2 in range(len(sentences)):
            if x1 != x2:
                similarity_matrix[x1][x2] = sentence_similarity(sentences[x1],sentences[x2], stop_words)
    return similarity_matrix

# Generate summary

def extractive_summary(text, top_n):
    
    nltk.download('stopwords')    
    nltk.download('punkt')

    stop_words = stopwords.words('english')
    summarize_text = []

    # read text and tokenize    
    sentences = read_article(text)
  # Generate similarity matrix            
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
  # Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
  # Sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
  
  # Get the top n number of sentences based on rank
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
  # Output the summarized version
    return " ".join(summarize_text),len(sentences)


def abstractive_summary(text):

    summariser = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    return summariser(text, min_length=5, max_length=20)


# Scrapping and Summarisation
'''
import urllib.request as rqst
from bs4 import BeautifulSoup
from string import punctuation
from nltk.corpus import stopwords

text = rqst.urlopen("https://en.wikipedia.org/wiki/Artificial_intelligence")
article = text.read()
article_parsed = BeautifulSoup(article, 'html.parser') 
paragraphs = article_parsed.find_all('p')
article_content = ""
for p in paragraphs:
    article_content += p.text
'''
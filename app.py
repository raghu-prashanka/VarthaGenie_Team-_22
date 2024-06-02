
import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import spacy
from spacy.lang.te import Telugu
from indicnlp.tokenize import sentence_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
import networkx as nx

# Define the Telugu NLP model
nlp = Telugu()

def scrape_h1_and_p_tags(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        h1_tag = soup.find('h1')
        title = h1_tag.get_text().strip() if h1_tag else 'No title found'
        p_tags = soup.find_all('p')
        paragraphs = ' '.join(p.get_text().strip() for p in p_tags)
        return {
            'title': title,
            'content': paragraphs
        }
    else:
        st.error(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def text_summarizer(raw_text):
    docx = nlp(raw_text)
    stopwords = []  # No need for stopwords in Telugu

    word_frequencies = {}
    sentences = sentence_tokenize.sentence_split(raw_text, lang='te')

    for sentence in sentences:
        for word in sentence.split():
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    c = CountVectorizer()
    bow_matrix = c.fit_transform(sentences)
    tfidf_transformer = TfidfTransformer()
    normalized_matrix = tfidf_transformer.fit_transform(bow_matrix)
    normalized_csr_matrix = csr_matrix(normalized_matrix)
    nx_graph = nx.Graph()

    for i, sentence in enumerate(sentences):
        nx_graph.add_node(i, sentence=sentence)

    for i in range(normalized_csr_matrix.shape[0]):
        for j in range(normalized_csr_matrix.shape[1]):
            weight = normalized_csr_matrix[i, j]
            if weight > 0:
                nx_graph.add_edge(i, j, weight=weight)

    scores = nx.pagerank(nx_graph)
    sorted_sentences = sorted(scores, key=scores.get, reverse=True)
    
    # Determine the number of sentences for the summary based on content length
    content_length = len(raw_text)
    if content_length < 500:
        num_sentences = 1
    elif 500 <= content_length < 1500:
        num_sentences = 3
    elif 1500 <= content_length < 3000:
        num_sentences = 4
    else:
        num_sentences = 5
    
    summary = ' '.join([sentences[i] for i in sorted_sentences[:num_sentences]])
    return summary

# Streamlit interface
st.title("Telugu Text Summarizer")

url = st.text_input("Enter the URL to scrape and summarize")

if url:
    scraped_data = scrape_h1_and_p_tags(url)
    if scraped_data:
        st.write("### Title")
        st.write(scraped_data['title'])
        
        st.write("### Content")
        st.write(scraped_data['content'])
        
        summary = text_summarizer(scraped_data['content'])
        
        st.write("### Summary")
        st.write(summary)

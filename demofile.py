import gradio as gr
import torch
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from pypdf import PdfReader
from io import BytesIO
import tempfile
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def choose_summary(file, method):
    if method == "BART":
        return generate_summary_bart(file)
    elif method == "Alternate Method":
        return generate_summary_alternate(file)



def read_pdf(file):
    # Write the bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file)
        temp_filename = temp.name

    # Create a PDF reader object
    pdf_reader = PdfReader(temp_filename)
    num_pages = len(pdf_reader.pages)

    # Loop through all the pages in the PDF document and extract the text
    text = ""
    for i in range(num_pages):
        # Get the page object
        page = pdf_reader.pages[i]

        # Extract the text from the page
        page_text = page.extract_text()

        # Append the text to the document text
        text += page_text

    # Define a regular expression to match the word "References" and everything that follows it
    references_pattern = re.compile(r'References(.*)', re.DOTALL)

    # Use the regular expression to remove the text after "References"
    text = references_pattern.sub('', text)

    

    updated_text = re.sub(r".*Abstract[^:]+:", "", text, flags=re.DOTALL)

    return updated_text



def generate_summary_bart(file):
    # Load the BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('./bert-large-tokenizer')

    # Load the saved model
    model = BartForConditionalGeneration.from_pretrained("model")  # replace "model" with the path to your .bin file if necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Read the uploaded PDF file
    input_text = read_pdf(file)
    # Generate the summary
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=1024)
    summary_ids = model.generate(input_ids, max_length=1024, num_beams=8)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def generate_summary_alternate(file):
    # Define a regular expression to match equations
    equations_pattern = re.compile(r'.*?=.+')

    # Perform word tokenization and create a frequency table
    text = read_pdf(file)
    words = word_tokenize(text.lower())
    freq_table = FreqDist(words)
    # Use the regular expression to remove sentences containing equations
    sentences = sent_tokenize(text)
    filtered_sentences = []
    for sentence in sentences:
        if not equations_pattern.search(sentence):
            filtered_sentences.append(sentence)
    sentences = filtered_sentences
    # Join the cleaned sentences into a single string
    text = " ".join(sentences)
    # Assign weights to each sentence and calculate sentence scores
    sentence_scores = {}
    relevant_keywords = ['neural', 'network', 'technique', 'algorithm', 'approach', 'model', 'method', 'function', 'problem']
    for sentence in sentences:
        sentence_score = 0
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                if len(sentence.split(" ")) < 30:
                    if sentence not in sentence_scores:
                        sentence_score += freq_table[word]
                    else:
                        sentence_score += freq_table[word]
                if word in relevant_keywords:
                    sentence_score += freq_table[word] * 2 # Give higher weight to relevant keywords
        sentence_scores[sentence] = sentence_score

    # Vectorize the sentences using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    # Cluster the sentences using KMeans algorithm
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)

    # Get the cluster labels
    labels = km.labels_

    # Create a dictionary to store the sentences in each cluster
    cluster_sentences = {}
    for i in range(num_clusters):
        cluster_sentences[i] = []

    # Assign each sentence to a cluster
    for i, label in enumerate(labels):
        cluster_sentences[label].append(sentences[i])

    # Get the top sentences in each cluster based on sentence scores
    top_sentences = []
    for i in range(num_clusters):
        sentences_in_cluster = cluster_sentences[i]
        sentences_in_cluster = [s for s in sentences_in_cluster if sentence_scores.get(s) is not None]  # Remove None scores
        top_sentences_in_cluster = nlargest(2, sentences_in_cluster, key=sentence_scores.get)
        top_sentences += top_sentences_in_cluster

    # Concatenate the selected sentences to produce the summary/abstract
    summary = " ".join(top_sentences)
    return summary




demo = gr.Interface(
    fn=choose_summary, 
    inputs=[gr.inputs.File(type="bytes"), gr.inputs.Radio(["BART", "Alternate Method"])], 
    outputs="text",
    enable_queue=True
)
demo.launch()


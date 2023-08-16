import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from flask import Flask, render_template, request, jsonify
from spacy.matcher import Matcher
import spacy
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PageObject, PdfReader
import networkx as nx
import urllib.parse
from textblob import TextBlob

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', results=[])

    if request.method == 'POST':
        print('Received POST request')

    urls = request.form.get('url')
    print('URLs:', urls)
    title = request.form.get('title')
    print('Title:', title)
    authors_input = request.form.get('author')
    authors = [author.strip() for author in authors_input.split(",")]
    print('Authors:', authors)

    custom_entities_input = request.form.get('customEntities')
    custom_entities = custom_entities_input.split(',') if custom_entities_input else []
    print('Custom entities:', custom_entities)

    only_custom_entities = request.form.get('onlyCustomEntities') == 'true'
    urls = [url.strip() for url in urls.split(",")]
    print('URLs list:', urls)

    results = []

    for i, url in enumerate(urls):
        if i < len(authors):
            author = authors[i]
        else:
            author = authors[-1]

        result = process_url(url, title, author, custom_entities, only_custom_entities)
        results.append(result)

    return jsonify(results)

@app.route('/delete_rows', methods=['POST'])
def delete_rows():
    data = request.json.get('data')
    indices_to_delete = request.json.get('indicesToDelete')
    for index in sorted(indices_to_delete, reverse=True):
        del data[index]

    return jsonify(data)

def process_url(url, title, author, custom_entities, only_custom_entities):
    print('Inside process_url', url)

    nlp = spacy.load('en_core_web_lg')
    matcher = Matcher(nlp.vocab)

    if custom_entities:
        patterns = [[{"TEXT": word} for word in entity.split()] for entity in custom_entities]
        matcher.add("CustomEntities", patterns)
    else:
        print('No custom entities provided.')

    url = urllib.parse.unquote(url)
    parsed_url = urllib.parse.urlparse(url)
    if not bool(parsed_url.scheme):
        url = "http://" + url

    try:
        result = urllib.parse.urlparse(url)
        is_valid = all([result.scheme, result.netloc, result.path])
    except ValueError as e:
        print(f"Invalid URL: {e}")
        return {"error": f"Invalid URL: {e}"}

    if not is_valid:
        print("Invalid URL.")
        return {"error": "Invalid URL."}

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return {"error": f"HTTP error occurred: {err}"}

    if response.headers['Content-Type'] == 'application/pdf':
        try:
            text = read_pdf(BytesIO(response.content))
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return {"error": f"Error reading PDF: {e}"}
    else:
        text = BeautifulSoup(response.content, 'html.parser').get_text()

    doc = nlp(text)

    entities = []

    if custom_entities or only_custom_entities:
        if custom_entities:
            matches = matcher(doc)
            entities += [(doc[start:end].text, "CUSTOM", get_entity_context(doc, start, end), get_entity_sentiment(get_entity_context(doc, start, end))) for match_id, start, end in matches]
    if not only_custom_entities:
        entities += [(ent.text, ent.label_, get_entity_context(doc, ent.start, ent.end), get_entity_sentiment(get_entity_context(doc, ent.start, ent.end))) for ent in doc.ents]

    return {"url": url, "title": title, "author": author, "entities": entities}

def read_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_entity_context(doc, start, end):
    sent_list = list(doc.sents)
    for i in range(len(sent_list)):
        if start >= sent_list[i].start and end <= sent_list[i].end:
            return sent_list[i].text
    return ""

def get_entity_sentiment(context):
    sentiment = TextBlob(context).sentiment.polarity
    return sentiment

if __name__ == '__main__':
    app.run(debug=True)

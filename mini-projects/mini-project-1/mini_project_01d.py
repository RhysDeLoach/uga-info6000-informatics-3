###############################################################################
# File Name: mini_project_01d.py
#
# Description: This Flask script implements a physics question-answering API 
# that takes a student's question, retrieves relevant context (RAG) from a 
# preloaded physics manual using TF-IDF cosine similarity, and generates a 
# formal, high-school-level explanation using a Flan-T5 language model. It 
# handles questions both within and outside the manual, synthesizing clear 
# answers while maintaining scientific terminology.
#
# Record of Revisions (Date | Author | Change):
# 09/21/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

# Load Data
with open("data/physics_context.json", "r") as f:
    manual = json.load(f)

# Extract contexts
documents = [entry["context"] for entry in manual]

# Preprocess Text
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

documents = [preprocess(doc) for doc in documents]

# Build TF-IDF TDM
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tdm = vectorizer.fit_transform(documents)

# Cosine Similarity Function with threshold
SIMILARITY_THRESHOLD = 0.15

def similarity(question, tdm, vectorizer, top_n=5):
    q_vec = vectorizer.transform([preprocess(question)])
    similarities = cosine_similarity(q_vec, tdm)
    top_score = similarities.max()
    if top_score < SIMILARITY_THRESHOLD:
        return []
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    return [documents[i] for i in top_indices]

# Set device
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Initialize Model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def answerQuestion(question, top_n=5):
    relevantContext = similarity(question, tdm, vectorizer, top_n=top_n)
    
    if relevantContext:
        context = " ".join(relevantContext)
        prompt = f"""
        You are a physics expert writing for a high school textbook. 
        Based on the context, provide a clear, formal explanation to the student's question. 
        Write in full sentences, use proper scientific terminology, and avoid listing numbers or choices.
        Do not just copy the context word for word; instead, synthesize the information to create a comprehensive answer.

        Context:
        {context}

        Question:
        {question}

        Well-written answer:
        """
    else:
        prompt = f"""
        You are a physics expert writing for a high school textbook. 
        The following question is not covered in the reference material:
        
        Question:
        {question}

        Provide a clear, formal explanation using general physics knowledge. Write in full sentences and use proper scientific terminology.
        """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        question = request.json['question']
        answer = answerQuestion(question, top_n=5)
        print(answer)
        return jsonify({'prediction': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

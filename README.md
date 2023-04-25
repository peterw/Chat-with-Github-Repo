# Chat-with-Github-Repo

This repository contains two Python scripts that demonstrate how to create a chatbot using Streamlit, OpenAI GPT-3.5-turbo, and DeepLake.

The chatbot searches a dataset stored in DeepLake to find relevant information and generates responses based on the user's input.

Files
github.py: This script clones a git repository, processes the text documents, computes embeddings using OpenAIEmbeddings, and stores the embeddings in a DeepLake instance.

chat.py: This script creates a Streamlit web application that interacts with the user and the DeepLake instance to generate chatbot responses using OpenAI GPT-3.5-turbo.

Setup
To set up and run this project, follow these steps:

Run the github.py script to embed the github repo first 

This is how you run the streamlit app: 
streamlit run chat.py

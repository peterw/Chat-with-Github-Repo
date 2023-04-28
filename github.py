import os
import subprocess
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')


def clone_repository(repo_url, local_path):
    subprocess.run(["git", "clone", repo_url, local_path])


def load_docs(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(
                    dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def main(repo_url, root_dir, deep_lake_path):
    clone_repository(repo_url, root_dir)
    docs = load_docs(root_dir)
    texts = split_docs(docs)
    embeddings = OpenAIEmbeddings()
    
    db = DeepLake(dataset_path=deep_lake_path, embedding_function=embeddings)
    
    db.add_documents(texts)


if __name__ == "__main__":
    repo_url = os.environ.get('REPO_URL')
    root_dir = "./gumroad"
    deep_lake_path = os.environ.get('DEEPLAKE_DATASET_PATH')

    main(repo_url, root_dir, deep_lake_path)

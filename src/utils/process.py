import os
import subprocess
import deeplake
import openai
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake

# Set the OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


def clone_repository(repo_url, local_path):
    """Clone the specified git repository to the given local path."""
    subprocess.run(["git", "clone", repo_url, local_path])


def load_docs(root_dir, file_extensions=None):
    """
    Load documents from the specified root directory.
    Optionally filter by file extensions.
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file_extensions and os.path.splitext(file)[1] not in file_extensions:
                continue
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception:
                pass
    return docs


def split_docs(docs):
    """Split the input documents into smaller chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def create_deeplake_dataset(activeloop_dataset_path, activeloop_token):
    """Create an empty DeepLake dataset with the specified path and token."""
    ds = deeplake.empty(
        activeloop_dataset_path,
        token=activeloop_token,
        overwrite=True,
    )

    ds.create_tensor("ids")
    ds.create_tensor("metadata")
    ds.create_tensor("embedding")
    ds.create_tensor("text")


def process(
    repo_url, include_file_extensions, activeloop_dataset_path, repo_destination
):
    """
    Process a git repository by cloning it, filtering files, splitting documents,
    creating embeddings, and storing everything in a DeepLake dataset.
    """
    activeloop_token = os.getenv("ACTIVELOOP_TOKEN")

    create_deeplake_dataset(activeloop_dataset_path, activeloop_token)

    clone_repository(repo_url, repo_destination)

    docs = load_docs(repo_destination, include_file_extensions)
    texts = split_docs(docs)

    embeddings = OpenAIEmbeddings()

    db = DeepLake(dataset_path=activeloop_dataset_path, embedding_function=embeddings)
    db.add_documents(texts)

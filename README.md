# Chat-with-Github-Repo

This repository contains Python scripts that demonstrate how to create a chatbot using Streamlit, OpenAI GPT-3.5-turbo, and Activeloop's Deep Lake.

The chatbot searches a dataset stored in Deep Lake to find relevant information from any Git repository and generates responses based on the user's input.

## Files

- `src/utils/process.py`: This script clones a Git repository, processes the text documents, computes embeddings using OpenAIEmbeddings, and stores the embeddings in a DeepLake instance.

- `src/utils/chat.py`: This script creates a Streamlit web application that interacts with the user and the DeepLake instance to generate chatbot responses using OpenAI GPT-3.5-turbo.

- `src/main.py`: This script contains the command line interface (CLI) that allows you to run the chatbot application.

## Setup

Before getting started, be sure to sign up for an [Activeloop](https://www.activeloop.ai/) and [OpenAI](https://openai.com/) account and create API keys.

To set up and run this project, follow these steps:

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/peterw/Chat-with-Git-Repo.git
cd Chat-with-Git-Repo
```

2. Install the required packages with `pip`:

```bash
pip install -r requirements.txt
```

For development dependencies, you can install them using the following command:

```bash
pip install -r dev-requirements.txt
```

3. Set the environment variables:

Copy the `.env.example` file:

```bash
cp .env.example .env
```

Provide your API keys and username:

```
OPENAI_API_KEY=your_openai_api_key
ACTIVELOOP_TOKEN=your_activeloop_api_token
ACTIVELOOP_USERNAME=your_activeloop_username
```

4. Use the CLI to run the chatbot application. You can either process a Git repository or start the chat application using an existing dataset.

> For complete CLI instructions run `python src/main.py --help`

To process a Git repository, use the `process` subcommand:

```bash
python src/main.py process --repo-url https://github.com/username/repo_name
```

You can also specify additional options, such as file extensions to include while processing the repository, the name for the Activeloop dataset, or the destination to clone the repository:

```bash
python src/main.py process --repo-url https://github.com/username/repo_name --include-file-extensions .md .txt --activeloop-dataset-name my-dataset --repo-destination repos
```

To start the chat application using an existing dataset, use the `chat` subcommand:

```bash
python src/main.py chat --activeloop-dataset-name my-dataset
```

The Streamlit chat app will run, and you can interact with the chatbot at `http://localhost:8501` (or the next available port) to ask questions about the repository.

## Sponsors

✨ Learn to build projects like this one (early bird discount): [BuildFast Course](https://www.buildfastcourse.com/)

## License

[MIT License](LICENSE)

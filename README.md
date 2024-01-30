# Chat-with-Github-Repo

# Introduction
Large language models (LLMs) accomplish a remarkable level of language comprehension during their training process. It enables them to generate human-like text and creates powerful representations from textual data. We already covered leveraging LangChain to use LLMs for writing content with hands-on projects.

We will focus on using the language models for generating embeddings from corpora. The mentioned representation will power a chat application that can answer questions from any text by finding the closest data point to an inquiry. This project focuses on finding answers from a GitHub repository’s text files like .md and .txt. So, we will start by capturing data from a GitHub repository and converting it to embeddings. These embeddings will be saved on the Activeloop’s Deep Lake vector database for fast and easy access. The Deep Lake’s retriever object will find the related files based on the user’s query and provide them as the context to the model. Lastly, the model leverages the provided information to the best of its ability to answer the question.


## What is Deep Lake?
It is a vector database that offers multi-modality storage for all kinds of data (including but not limited to PDFs, Audio, and Videos) alongside their vectorized representations. This service eliminates the need to create data infrastructure while dealing with high-dimensionality tensors. Furthermore, it provides a wide range of functionalities like visualizing, parallel computation, data versioning, integration with major AI frameworks, and, most importantly, embedding search. The supported vector operations like cosine_similarity allow us to find relevant points in an embedding space.

The rest of the lesson is based on the code from the “Chat with Github Repo” repository and is organized as follows: 

1) Processing the Files 

2) Saving the Embedding 

3) Retrieving from Database 

4) Creating an Interface.

## Processing the Repository Files
In order to access the files in the target repository, the script will clone the desired repository onto your computer, placing the files in a folder named "repos". Once we download the files, it is a matter of looping through the directory to create a list of files. It is possible to filter out specific extensions or environmental items

```
root_dir = "./path/to/cloned/repository"
docs = []
file_extensions = []

for dirpath, dirnames, filenames in os.walk(root_dir):
	
	for file in filenames:
	  file_path = os.path.join(dirpath, file)
	
	  if file_extensions and os.path.splitext(file)[1] not in file_extensions:
      continue
	
    loader = TextLoader(file_path, encoding="utf-8")
    docs.extend(loader.load_and_split())
```

The sample code above creates a list of all the files in a repository. It is possible to filter each item by extension types like file_extensions=['.md', '.txt'] which only focus on markdown and text files. The original implementation has more filters and a fail-safe approach; Please refer to the complete code.

Now that the list of files are created, the split_documents method from the CharacterTextSplitter class in the LangChain library will read the files and split their contents into chunks of 1000 characters.

```
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_text = text_splitter.split_documents(docs)
```

The splitted_text variable holds the textual content which is ready to be converted to embedding representations.

## Saving the Embeddings
Let’s create the database before going through the process of converting texts to embeddings. It is where the integration between LangChain and Deep Lake comes in handy! We initialize the database in cloud using the hub://... format and the OpenAIEmbeddings() from LangChain as the embedding function. The Deep Lake library will iterate through the content and generate the embedding automatically.


```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "langchain_course_chat_with_gh"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(splitted_text)
```

## Retrieving from Database
The last step is to code the process to answer the user’s question based on the database’s information. Once again, the integration of LangChain and Deep Lake simplifies the process significantly, making it exceptionally easy. We need 1) a retriever object from the Deep Lake database using the .as_retriever() method, and 2) a conversational model like ChatGPT using the ChatOpenAI() class.

Finally, LangChain’s RetrievalQA class ties everything together! It uses the user’s input as the prompt while including the results from the database as the context. So, the ChatGPT model can find the correct one from the provided context. It is worth noting that the database retriever is configured to gather instances closely related to the user’s query by utilizing cosine similarities.

```
# Create a retriever from the DeepLake instance
retriever = db.as_retriever()

# Set the search parameters for the retriever
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 10

# Create a ChatOpenAI model instance
model = ChatOpenAI()

# Create a RetrievalQA instance from the model and retriever
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Return the result of the query
qa.run("What is the repository's name?")
```

## Create an Interface
Creating a user interface (UI) for the bot to be accessed through a web browser is an optional yet crucial step. This addition will elevate your ideas to new heights, allowing users to engage with the application effortlessly, even without any programming expertise. This repository uses the Streamlit platform, a fast and easy way to build and deploy an application instantly for free. It provides a wide range of widgets to eliminate the need for using backend or frontend frameworks to build a web application.

We must install the library and its chat component using the pip command. We strongly recommend installing the latest version of each library. Furthermore, the provided codes have been tested using streamlit and streamlit-chat versions 2023.6.21 and 20230314, respectively.

```
pip install streamlit streamlit_chat
```

The API documentation page provides a comprehensive list of available widgets that can use in your application. We need a simple UI that accepts the input from the user and shows the conversation in a chat-like interface. Luckily, Streamlit provides both.


```
import streamlit as st
from streamlit_chat import message

# Set the title for the Streamlit app
st.title(f"Chat with GitHub Repository")

# Initialize the session state for placeholder messages.
if "generated" not in st.session_state:
	st.session_state["generated"] = ["i am ready to help you ser"]

if "past" not in st.session_state:
	st.session_state["past"] = ["hello"]

# A field input to receive user queries
input_text = st.text_input("", key="input")

# Search the databse and add the responses to state
if user_input:
	output = qa.run(user_input)
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

# Create the conversational UI using the previous states
if st.session_state["generated"]:
	for i in range(len(st.session_state["generated"])):
		message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
		message(st.session_state["generated"][i], key=str(i))
```

The code above is straightforward. We call st.text_input() to create text input for users queries. The query will be passed to the previously declared RetrievalQA object, and the results will be shown using the message component. You should store the mentioned code in a Python file (for example, chat.py) and run the following command to see the interface locally.

```
streamlit run ./chat.py
```


This repository contains Python scripts that demonstrate how to create a chatbot using Streamlit, OpenAI GPT-3.5-turbo, and Activeloop's Deep Lake.

the codes in are available in here “Chat with GitHub Repo,” you can easily fork and run it in 3 simple steps. First, fork the repository and install the required libraries using pip.

```
git clone https://github.com/peterw/Chat-with-Git-Repo.git
cd Chat-with-Git-Repo

pip install -r requirements.txt
```

Second, rename the environment file from .env.example to .env and fill in the API keys. You must have accounts in both OpenAI and Activeloop.

```
cp .env.example .env

# OPENAI_API_KEY=your_openai_api_key
# ACTIVELOOP_TOKEN=your_activeloop_api_token
# ACTIVELOOP_USERNAME=your_activeloop_username
```

Lastly, use the process command to read and store the contents of any repository on the Deep Lake by passing the repository URL to the --repo-url argument.

```
python src/main.py process --repo-url https://github.com/username/repo_name
```

Be aware of the costs associated with generating embeddings using the OpenAI API. Using a smaller repository that needs fewer resources and faster processing is better.

And run the chat interface by using the chat command followed by the database name. It is the same as repo_name from the above sample. You can also see the database name by logging in to the Deep Lake dashboard.
```
python src/main.py chat --activeloop-dataset-name <dataset_name>
```

The application will be accessible using a browser on the http://localhost:8501 URL or the next available port. (as demonstrated in the image below) Please read the complete instruction for more information, like filtering a repository content by file extension.


 

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

✨ Learn to build projects like this one (early bird discount): [BuildFast Course](https://www.buildfast.academy)

## License

[MIT License](LICENSE)

# Generative Chatbot Using Large Language Model

<h2>Summary</h2>

This project is for an externship to create a customer service chatbot. It is trained on data from Stack Overflow, specifically questions about Python. It uses models from SentenceTransformer and Llama2, then leverages ConversationalRetrievalChain in order to generate Question/Answer responses. The UI was built using Chainlit and deployed initially on Google Colab. After that, I created a Docker image and then deployed on AWS.


<h2>Table of Contents</h2>

      1. Exploratory Data Analysis(notebooks > EDA.ipynb)
            a) Package installation
            b) Library importation
            c) Import the data
            d) Data preprocessing - includes removing extraneous HTML tags
            e) Merging datasets into one dataframe
            f) EDA
            g) Data cleaning
            h) Create a new CSV
      
      2. Machine Learning Models (notebooks > ML.ipynb)
            
            a) Package installation
            b) Library importation
            c) Import the data
            d) Split the data
            e) Create vectors/embeddings
            f) Create the model
            g) Create the pipeline
            h) Query based on the trained model

      3. Application (app.py)
            a) Library Importations
            b)The initial part of this file (cl.on_chat_start) is the same as ML.ipynb
            c) The second part of this file (cl.on_message) is the part of the application where the chatbot responds to a question asked.
      

<h2>Local Access</h2>

This project will require Python 3.11.5 or later (if available).

All packages required can be installed from the requirements.txt file by executing the command 'pip install -r requirements.txt'

Please note that a HuggingFace API token, a Pinecone API token, and a Pinecone environment name will be required for this project to run correctly.

To run the app.py file and the Jupyter notebooks, a GPU must be used due to the size of the LLM and the size of the data. It is not recommended that this be run on a local machine.

To simulate running locally, Google Colab can be used with one of the GPU options.

<h2>Plans for Updates</h2>

There are no plans for updates at this time.

<h2>Sample Graph</h2>

![image](https://github.com/LDeYoung17/dataspeak-llm/assets/70500225/7dd6482c-4ad2-4c71-9876-df71c2234190)

<h2>Demonstration Video</h2>

https://github.com/LDeYoung17/dataspeak-llm/assets/70500225/54880afc-d58f-4bf8-a006-c6bb062cecb0


<h2>Portfolio Link</h2>

https://ldeyoung17.github.io/

This is my portfolio where all my projects, including this one, can be found, as well as more information about my experience as a Data Scientist and Software Engineer.

# dataspeak-llm

<h2>Summary</h2>

This project is for an externship to create a customer service chatbot. It is trained on data from Stack Overflow, specifically questions about Python. It uses a model from SentenceTransformer and then ConversationalRetrievalChain to generate Question/Answer responses.
<h2>Table of Contents</h2>

      1. Exploratory Data Analysis(notebooks > EDA.ipynb)
            a) Package installation
            b) Library importation
            c) Import the data
            d) Data preprocessing - includes removing extraneous HTML tags
            e) Merging datasets into one dataframe
            f) EDA
            g) Data cleaning
            h) Create new CSV
      
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

This project needs to be run on the GPU of your choice. The installations necessary are built into the project.

<h2>Plans for Updates</h2>

There are no plans for updates at this time.

<h2>Sample Graph</h2>

![image](https://github.com/LDeYoung17/dataspeak-llm/assets/70500225/7dd6482c-4ad2-4c71-9876-df71c2234190)

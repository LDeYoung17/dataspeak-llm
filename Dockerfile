FROM dataspeak-llm as builder



# Install chainlit and any other dependencies
RUN pip install chainlit
RUN pip install torch==2.1.0
RUN pip install --upgrade torchaudio torchdata torchtext torchvision
RUN pip install accelerate
RUN pip install evaluate
RUN pip install tensorflow
RUN pip install sentence-transformers
RUN pip install transformers==4.31.0
RUN pip install sentence-transformers==2.2.2
RUN pip install pinecone-client==2.2.2
RUN pip install datasets==2.14.0
RUN pip install accelerate==0.21.0
RUN pip install einops==0.6.1
RUN !pip install langchain==0.0.240
RUN pip install xformers==0.0.20
RUN pip install bitsandbytes==0.41.0

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY UI/app.py /app/app.py

ENV dataspeak-llm as runtime

RUN apk add --no-cache git
RUN git clone --branch main --single-branch https://github.com/LDeYoung17/dataspeak-llm.git /buildkit



EXPOSE 8080

# Start the app using chainlit run command
CMD ["chainlit", "run", "/app/app.py"]
import chainlit as cl
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
import torch
import tensorflow as tf
from torch import cuda, bfloat16
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
import time
import sys
import json
from my_package import secrets 


@cl.cache
@cl.on_chat_start
def create_storage():

    pinecone_api = secrets.get('PINECONE_API') 
    pinecone_env = secrets.get('PINECONE_ENVIRON')
    pinecone.init(
        api_key=os.environ.get(pinecone_api) or pinecone_api,
        environment=os.environ.get(pinecone_env) or pinecone_env
    )

    index_name = 'dataspeak-qa'


    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
    index = pinecone.Index(index_name)

async def start():

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    model_id = 'meta-llama/Llama-2-13b-chat-hf'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    hf_key = secrets.get('HUGGING_FACE_TOKEN')

    hf_auth = hf_key
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        #cache_dir = cache_dir,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    model.eval()

    #print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        #cache_dir=cache_dir,
        use_auth_token=hf_auth
    )

    stop_list = ['\nContext:', '\n```\n', '\nAnswer:']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.0,
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)

    text_field = 'text'

    vectorstore = Pinecone(
        create_storage.index, create_storage.embed_model.embed_query, text_field
    )

    query = 'tell me about python'

    search_result = retriever=vectorstore.similarity_search(query,
        k=5  # returns top 5 most relevant chunks of text
    )

    question_answer = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectorstore.as_retriever(k=5)
    )

    #cl.user_session.set("query_engine", question_answer)

    return question_answer

    
    
@cl.on_message
async def main(message: cl.Message):

    embeddings = await start() 

    #await cl.AskUserMessage(content="Welcome! How can I help you today?", timeout=30).send()

    #query_engine = cl.user_session.get("query_engine")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    response = await cl.make_async(embeddings.query)(message.content, callbacks=[cb])

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
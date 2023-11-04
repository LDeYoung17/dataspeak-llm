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
#from my_package import secrets 


@cl.on_chat_start
def start():
#cl.init()
  embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

  device = f'cuda:{cuda.current_device()}'
  #if cuda.is_available() else 'cpu'

  embed_model = HuggingFaceEmbeddings(
      model_name=embed_model_id,
      model_kwargs={'device': device},
      encode_kwargs={'batch_size': 32}
  )

  #pinecone_api = '5833daf1-6582-4254-92eb-367ec190841c'
  #pinecone_env = 'gcp-starter'
  pinecone.init(
      api_key=os.environ.get('PINECONE_API'),
      environment=os.environ.get('PINECONE_ENVIRON')
  )

  index_name = 'dataspeak-qa'


  while not pinecone.describe_index(index_name).status['ready']:
      time.sleep(1)
  index = pinecone.Index(index_name)

  model_id = 'meta-llama/Llama-2-13b-chat-hf'

  device = f'cuda:{cuda.current_device()}'
  #if cuda.is_available() else 'cpu'
  bnb_config = transformers.BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_use_double_quant=True,
      #llm_int8_enable_fp32_cpu_offload=True,
      bnb_4bit_compute_dtype=bfloat16
  )


  hf_auth = os.environ.get('HUGGING_FACE_TOKEN')
  model_config = transformers.AutoConfig.from_pretrained(
      model_id,
      use_auth_token=hf_auth
  )

  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_id,
      trust_remote_code=True,
      config=model_config,
      quantization_config=bnb_config,
      device_map='auto',
      use_auth_token=hf_auth
  )

  model.eval()

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
              else:
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
      index,embed_model.embed_query, text_field
  )


  memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True, output_key='answer')

  retriever=vectorstore.as_retriever(k=5)
  chain = ConversationalRetrievalChain.from_llm(llm, chain_type="stuff", retriever=retriever, memory=memory, return_source_documents=True)

  question_answer = RetrievalQA.from_chain_type(
      llm=llm, chain_type='stuff',
      retriever=vectorstore.as_retriever(k=5), memory=memory
      #, return_source_documents=True
  )

  cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
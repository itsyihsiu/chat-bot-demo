import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import chainlit as cl
from utils import *

# pip install chainlit langchain google-generativeai langchain-google-genai
# pip install chromadb langchain-community sentence-transformers openai
# python docs.py
# chainlit run main.py -w

os.environ["GOOGLE_API_KEY"]="your_google_api_key_here"

llm = ChatGoogleGenerativeAI(model="gemini-pro")

template = """<s>[INST] You are a helpful, respectful and honest assistant. 
Please answer the question below in Traditional Chinese.

Question: {question} [/INST] </s>
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm | StrOutputParser()

# output = chain.invoke({"question": "興農公司董事長是誰"})
# print(f"{output}")

template = """
<s>[INST] You are a helpful, respectful and honest assistant.
Please response with Traditional Chinese. Answer the question based on your knowledge. Here is context to help:

{context}

Question: {question} [/INST] </s>
"""

rag_prompt = PromptTemplate.from_template(template)
db = Chroma(persist_directory="./chromadb", embedding_function=HuggingFaceEmbeddings())
# db = Chroma(persist_directory="./chromadb", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = db.as_retriever(max_tokens_limit=512, search_kwargs={"k": 2})
rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

# output = rag_chain.invoke("興農公司總經理")
# print(f"{output}")

@cl.on_message
async def main(message: cl.Message):

    # output = chain.invoke({"question": message.content})
    output = rag_chain.invoke(message.content)

    print(f"\n{output}")

    await cl.Message(
        content=f"{output}",
    ).send()
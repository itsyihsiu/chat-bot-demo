import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import chainlit as cl

# pip install --upgrade chainlit langchain google-generativeai langchain-google-genai
# chainlit run main.py -w --port $(((RANDOM%999) + 1024))

os.environ["GOOGLE_API_KEY"]="your_google_api_key"

llm = ChatGoogleGenerativeAI(model="gemini-pro")

template = """<s>[INST] You are a helpful, respectful and honest assistant. 
Please answer the question below in Traditional Chinese.

Question: {question} [/INST] </s>
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm | StrOutputParser()

@cl.on_message
async def main(message: cl.Message):

    output = chain.invoke({"question": message.content})

    print(f"\n{output}")

    await cl.Message(
        content=f"{output}",
    ).send()
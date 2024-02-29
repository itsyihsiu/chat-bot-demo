import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.errors import InvalidDimensionException
from utils import *

os.environ["GOOGLE_API_KEY"]="your_google_api_key_here"

loaders = [
    CSVLoader(file_path = './qa.csv', encoding = 'UTF-16'),
]

documents = []

for loader in loaders:
    documents.extend(loader.load())

# embedding = HuggingFaceEmbeddings()
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

Chroma().delete_collection()

try:
    db = Chroma.from_documents(documents, embedding=embedding, persist_directory='./chromadb')
except InvalidDimensionException:
    Chroma().delete_collection()
    db = Chroma.from_documents(documents, embedding=embedding, persist_directory='./chromadb')

if __name__ == '__main__':
    retriever = db.as_retriever(max_tokens_limit=4096, search_kwargs={"k": 5})
    answer = '\n\n' + format_docs(retriever.invoke('董事長'))
    print(answer)
   



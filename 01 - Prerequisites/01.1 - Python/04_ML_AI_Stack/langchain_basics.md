# LangChain Basics

## LLM Call
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")
response = llm.invoke("Explain RAG in one sentence")
print(response.content)
```

## Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])
chain = prompt | llm
chain.invoke({"question": "What is PyTorch?"})
```

## Simple RAG
```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

combine_docs = create_stuff_documents_chain(llm, prompt)
rag = create_retrieval_chain(retriever, combine_docs)
result = rag.invoke({"input": "question here"})
```

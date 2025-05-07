from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
    num_predict=500
)

from langchain_ollama import OllamaEmbeddings
local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


#load the PDF File using PyPDFLoader
data_loader = PyPDFLoader("D:\HybridRAG using OLLAMA\Atomic habits ( PDFDrive ).pdf")
data = data_loader.load()

#splitting the data using Recursive-Character-TextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = text_splitter.split_documents(data)


#store the embedded data into FAISS vectorDB
vector_store = FAISS.from_documents(documents=chunks,embedding=local_embeddings)

vectorstore_retriever = vector_store.as_retriever(search_kwargs={"K":3})

#keyword retriever
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k=3

#using Ensemble Retriever
ensemble_result = EnsembleRetriever(retrievers=[vectorstore_retriever,keyword_retriever],
                                   weights=[0.5,0.5])

template = """
<|system|>>
You are a helpful AI Assistant that follows instructions extremely well.
Use the following context to answer user question.

Think step by step before answering the question. You will get a $100 tip if you provide correct answer.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": ensemble_result, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

print(chain.invoke("Can you explain First law of Atomic Habits in a detailed manner?"))
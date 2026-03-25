from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from typing import List
from langchain.schema import Document

# Extract data from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



# Filter the documents to only include the minimal set of documents needed to answer the question.
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter the documents to only include the minimal set of documents needed to answer the question.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src  = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src             
                }
            )
        )
    return minimal_docs



# split the documents into chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,

    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# Download the embeddings model from HuggingFace
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings
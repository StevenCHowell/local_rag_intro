import logging
import os
import pathlib
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(pdf_directory: str | pathlib.Path) -> list[Document]:
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
    documents (list[Document]): List of Document objects containing text content to split.
    Returns:
    list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    logger.debug(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print example of page content and metadata for a chunk
    document = chunks[0]
    logger.debug(f"First chunk content: {document.page_content}")
    logger.debug(f"First chunk metadata: {document.metadata}")

    return chunks  # Return the list of split text chunks


# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"


def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """

    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store


def main():
    logger.info("Hello from local-rag-intro!")

    pdf_directory = pathlib.Path("data/pdf")
    documents = load_documents(pdf_directory)
    logger.debug(f"Loaded {len(documents)} documents from {pdf_directory}")

    chunks = split_text(documents)
    logger.debug(f"Split documents into {len(chunks)} chunks.")

    save_to_chroma(chunks)

    # Load environment variables from a .env file
    load_dotenv()

    # Generate the data store
    generate_data_store()


if __name__ == "__main__":

    main()

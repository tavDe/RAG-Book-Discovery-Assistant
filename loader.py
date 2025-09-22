from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document

class BooksLoader:
    def __init__(self, path):
        """Initialise file path."""
        self.path = path

    def document_load(self) -> List[Document]:
        """Load articles from data/books.txt Text file.
        - Must return documents.
        - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} not found")
        loader = TextLoader(self.path)
        return loader.load()

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """split the documents into chunks of size 500 and overlap of 50. Returns the created chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(documents)

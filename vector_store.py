from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

class BooksVectorStore:
    def __init__(self):
        """Initialize vector store with OpenAI embeddings."""
        self.embeddings = None
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store from documents."""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents.

        Raises:
            ValueError: If vector_store is not initialized.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.similarity_search(query, k=k)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

from vector_store import BooksVectorStore

class BooksRAGChain:
    def __init__(self, vector_store: BooksVectorStore):
        """Initialize RAG chain with vector store and OpenAI embeddings."""
        self.vector_store = vector_store
        self.chain = None
        self._create_chain()

    def _create_chain(self):
        """Create the RAG chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that makes book recommendations based on the following context."),
                ("human", "Query: {question}\nRelevant Extracts: {context}"),
            ]
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)  # Use a valid model for your keys/access.
        output_parser = StrOutputParser()
        self.chain = prompt | llm | output_parser

    async def query(self, question: str) -> str:
        """Query the RAG chain and return the response"""
        relevant_docs = self.get_relevant_documents(question, k=5)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        input_data = {"question": question, "context": context}
        return await self.chain.ainvoke(input_data)

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents with metadata.
        Raises:
            ValueError: If query is empty or too short (Less than 10 characters)
        """
        if not query or len(query) < 10:
            raise ValueError("Query too short.")
        return self.vector_store.similarity_search(query, k=k)

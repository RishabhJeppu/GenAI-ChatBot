import os
import sys
import torch
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../",
            "venv/lib/site-packages",
        )
    )
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


class KnowledgeBase:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant = QdrantClient(path="./")
        self.collection_name = "gen_ai_data"
        self.file_path = (
            "../raw_data/Comprehensive Guide to GenAI Application Development.txt"
        )

        # Initialize Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a given text using Hugging Face models.
        """
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert to numpy array and ensure correct shape
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Handle single dimension case
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def load_and_split_documents(self):
        """
        Load the document and split into chunks using RecursiveCharacterTextSplitter.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                full_text = file.read()
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_text(full_text)
        return chunks

    def store_documents(self):
        """
        Store document chunks in existing Qdrant collection.
        """
        documents = self.load_and_split_documents()
        if not documents:
            print("No documents to store.")
            return

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            points = []

            for idx, doc in enumerate(batch):
                embedding = self.embed_text(doc)
                points.append(
                    PointStruct(
                        id=i + idx,
                        vector=embedding[0].tolist(),  # Just use vector field
                        payload={"text": doc},
                    )
                )

            try:
                self.qdrant.delete_collection(collection_name=self.collection_name)
                print(f"Deleted existing collection '{self.collection_name}'")
            except Exception as e:
                print(f"Collection '{self.collection_name}' doesn't exist yet")

            # Create new collection
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": 384,
                    "distance": "Cosine",
                },
            )
            print(f"Created new collection '{self.collection_name}'")

            # Upload points
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"Batch {i // batch_size + 1} stored successfully!")

        print("All documents stored successfully in Qdrant!")

    def retrieve_documents(self, query: str, limit: int = 5):
        """
        Search for similar documents based on a query.
        """
        query_embedding = self.embed_text(query)

        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=limit,
            with_payload=True,
        )
        return search_result

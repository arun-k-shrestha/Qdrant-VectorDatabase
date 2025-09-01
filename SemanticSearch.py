from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Load documents from a text file
with open("Dataset1.txt", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load pre-trained sentence transformer model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# This tells Qdrant where to store embeddings. Local computer will use its memory as temporary storage.
client = QdrantClient(":memory:")

# Create a collection named "my_books" with vector configuration
client.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# Upload documents with their embeddings to the Qdrant collection
client.upload_points(
    collection_name="my_books",
    points = [
        models.PointStruct(
            id =idx,
            vector=encoder.encode(doc["description"]).tolist(),payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

print(documents)


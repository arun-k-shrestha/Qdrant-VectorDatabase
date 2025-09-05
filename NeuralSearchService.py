from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Load the dataset
df = pd.read_json('./startups_demo.json',lines=True)

# Generate embeddings for the 'alt' and 'description' fields
vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()],
    show_progress_bar=True,
)

# Save the embeddings to a .npy file. Here, .npy is a binary file format used by NumPy to store arrays efficiently
np.save('startup_vectors.npy', vectors,allow_pickle=False)
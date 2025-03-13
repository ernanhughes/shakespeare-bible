from tqdm import tqdm
import chromadb
import sqlite3
import subprocess
import json
import ollama
import numpy as np

# Initialize ChromaDB
client = chromadb.PersistentClient(path="literature_chroma_db")
bible_collection = client.get_or_create_collection(name="bible_verses")

# Function to get embeddings using Ollama (with error handling)
def get_embedding(text):
    try:
        embedding_data = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding_data = embedding_data["embedding"]  # Extract embedding
        embedding_data = np.array(embedding_data)
        return embedding_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error for text: {text[:50]}... - {e}")
    except subprocess.SubprocessError as e:
        print(f"❌ Error running Ollama subprocess: {e}")
    except Exception as e:
        print(f"❌ Unexpected error generating embedding: {e}")

    return None  # Return None if an error occurs

# Load Bible verses into ChromaDB in batches
def load_bible_into_chroma(batch_size=100):
    conn = sqlite3.connect("bible.db")
    cursor = conn.cursor()
    cursor.execute("SELECT verse, text FROM bible_verses")
    verses = cursor.fetchall()
    
    batch_ids, batch_embeddings, batch_metadatas = [], [], []
    
    for verse_ref, verse_text in tqdm(verses, desc="Loading Bible Verses into ChromaDB", unit="verse"):
        try:
            embedding = get_embedding(verse_text)
            if embedding.any():  # Skip if embedding failed
                batch_ids.append(verse_ref)
                batch_embeddings.append(embedding)
                batch_metadatas.append({"verse": verse_ref, "text": verse_text})

                # When batch size reaches `batch_size`, insert into ChromaDB
                if len(batch_ids) >= batch_size:
                    bible_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )
                    batch_ids, batch_embeddings, batch_metadatas = [], [], []  # Reset batch lists

        except chromadb.errors.ChromaError as e:
            print(f"❌ ChromaDB error for verse {verse_ref}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error storing verse {verse_ref} in ChromaDB: {e}")

    # Insert remaining verses (if any) after loop completes
    if batch_ids:
        bible_collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

    conn.close()
    print("✅ Bible verses successfully loaded into ChromaDB in batches!")

# Run the process
load_bible_into_chroma(batch_size=100)

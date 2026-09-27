import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000

def run_c1():
    print("=== [C1] Inserción de Embeddings Precalculados en ChromaDB ===")
    
    # 1. Cargar oraciones desde sentences.txt
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones desde 'sentences.txt'.")

    # 2. Generar embeddings con SentenceTransformer
    print("Cargando el modelo 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Generando embeddings...")
    start_emb = time.perf_counter()
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    end_emb = time.perf_counter()
    print(f"Embeddings generados en: {end_emb - start_emb:.4f} s.")

    # 3. Inicializar cliente persistente de Chroma
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name="c1_sentences_vectors")
    except Exception:
        pass

    collection = client.create_collection(
        name="c1_sentences_vectors",
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Inserción por lotes midiendo el tiempo de almacenamiento
    storage_times = []
    
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk_docs = sentences[i : i + CHUNK_SIZE]
        chunk_embs = embeddings[i : i + CHUNK_SIZE].tolist()
        chunk_ids = [f"vec_{j}" for j in range(i, i + len(chunk_docs))]
        
        t0 = time.perf_counter()
        collection.add(
            embeddings=chunk_embs,
            documents=chunk_docs,
            ids=chunk_ids
        )
        t1 = time.perf_counter()
        
        storage_times.append(t1 - t0)

    # 5. Cálculo de métricas estadísticas de almacenamiento de embeddings
    min_time = min(storage_times)
    max_time = max(storage_times)
    avg_time = statistics.mean(storage_times)
    std_time = statistics.stdev(storage_times) if len(storage_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Embeddings [C1] ({len(storage_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")

if __name__ == "__main__":
    run_c1()
import time
import statistics
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from config import load_config
from connect import connect

CHUNK_SIZE = 1000

def run_g1():
    print("=== [G1] Inserción de Embeddings con pgvector ===")
    
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones.")
    print("Generando embeddings con 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)

    config = load_config()
    conn = connect(config)
    if conn is None:
        return

    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("DROP TABLE IF EXISTS g1_sentences_vectors;")
    cursor.execute("""
        CREATE TABLE g1_sentences_vectors (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector(384) NOT NULL
        );
    """)
    conn.commit()

    # Formatear la lista de tuplas para la consulta masiva
    data = [(text, str(emb.tolist())) for text, emb in zip(sentences, embeddings)]
    query = "INSERT INTO g1_sentences_vectors (text, embedding) VALUES %s"
    template = "(%s, %s::vector)"
    
    storage_times = []

    # Inserción en bloques estrictos de CHUNK_SIZE (1000)
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        
        t0 = time.perf_counter()
        execute_values(cursor, query, chunk, template=template)
        conn.commit()
        t1 = time.perf_counter()
        
        storage_times.append(t1 - t0)

    # Métricas de rendimiento
    min_time = min(storage_times)
    max_time = max(storage_times)
    avg_time = statistics.mean(storage_times)
    std_time = statistics.stdev(storage_times) if len(storage_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Embeddings [G1 - pgvector] ({len(storage_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_g1()
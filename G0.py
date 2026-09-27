import time
import statistics
from psycopg2.extras import execute_values
from config import load_config
from connect import connect

CHUNK_SIZE = 1000

def run_g0():
    print("=== [G0] Carga de Texto Plano en PostgreSQL (pgvector) ===")
    
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones desde 'sentences.txt'.")

    config = load_config()
    conn = connect(config)
    if conn is None:
        return

    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cursor.execute("DROP TABLE IF EXISTS g0_sentences;")
    cursor.execute("""
        CREATE TABLE g0_sentences (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)
    conn.commit()

    query = "INSERT INTO g0_sentences (text) VALUES %s"
    chunk_times = []

    # Inserción en bloques estrictos de CHUNK_SIZE (1000)
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk = [(s,) for s in sentences[i : i + CHUNK_SIZE]]
        
        t0 = time.perf_counter()
        execute_values(cursor, query, chunk)
        conn.commit()
        t1 = time.perf_counter()
        
        chunk_times.append(t1 - t0)

    # Métricas de rendimiento
    min_time = min(chunk_times)
    max_time = max(chunk_times)
    avg_time = statistics.mean(chunk_times)
    std_time = statistics.stdev(chunk_times) if len(chunk_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Texto [G0] ({len(chunk_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_g0()
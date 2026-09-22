import time
import statistics
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from config import load_config
from connect import connect

# CONFIGURACIÓN DEL TAMAÑO DE CHUNK
CHUNK_SIZE = 1000

def run_p1():
    print("=== [P1] Generación e Inserción de Embeddings en PostgreSQL ===")
    
    # 1. Cargar oraciones desde sentences.txt
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones desde 'sentences.txt'.")

    # 2. Cargar el modelo transformer y generar vectores
    print("Cargando el modelo 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    start_emb_time = time.perf_counter()
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    end_emb_time = time.perf_counter()
    emb_duration = end_emb_time - start_emb_time
    print(f"Embeddings generados en: {emb_duration:.4f} segundos.")

    # 3. Conectar a PostgreSQL usando config.py y connect.py
    config = load_config()
    conn = connect(config)
    
    if conn is None:
        print("Error: No se pudo establecer la conexión a PostgreSQL.")
        return

    cursor = conn.cursor()

    # 4. Crear la tabla relacional con soporte para vectores flotantes
    cursor.execute("DROP TABLE IF EXISTS p1_sentences_vectors;")
    cursor.execute("""
        CREATE TABLE p1_sentences_vectors (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL
        );
    """)
    conn.commit()

    # 5. Insertar datos y vectores en chunks y medir tiempo de BD
    print(f"Insertando en PostgreSQL en lotes de {CHUNK_SIZE} registros...")
    
    data = [(text, emb.tolist()) for text, emb in zip(sentences, embeddings)]
    query = "INSERT INTO p1_sentences_vectors (text, embedding) VALUES %s"
    storage_times = []

    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        
        t0 = time.perf_counter()
        execute_values(cursor, query, chunk)
        conn.commit()
        t1 = time.perf_counter()
        
        storage_times.append(t1 - t0)

    
    # 6. Cálculo de métricas sobre el tiempo de almacenamiento
    min_time = min(storage_times)
    max_time = max(storage_times)
    avg_time = statistics.mean(storage_times)
    std_time = statistics.stdev(storage_times) if len(storage_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Embeddings [P1] ({len(storage_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_p1()
import time
import statistics
from psycopg2.extras import execute_values
from config import load_config
from connect import connect

# CONFIGURACIÓN DEL TAMAÑO DE CHUNK
CHUNK_SIZE = 1000

def run_p0():
    print("=== [P0] Carga de Texto Plano en PostgreSQL ===")
    
    # 1. Cargar oraciones desde sentences.txt
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones desde 'sentences.txt'.")
    print(f"Tamaño de lote (CHUNK_SIZE) configurado: {CHUNK_SIZE}")

    # 2. Obtener configuración y conectar usando connect.py
    config = load_config()
    conn = connect(config)
    
    if conn is None:
        print("Error: No se pudo establecer la conexión a PostgreSQL.")
        return

    cursor = conn.cursor()

    # 3. Crear la tabla relacional básica
    cursor.execute("DROP TABLE IF EXISTS p0_sentences;")
    cursor.execute("""
        CREATE TABLE p0_sentences (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)
    conn.commit()

    # 4. Insertar textos en chunks a la BD y medir tiempos
    query = "INSERT INTO p0_sentences (text) VALUES %s"
    chunk_times = []
    
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk = [(s,) for s in sentences[i : i + CHUNK_SIZE]]
        
        t0 = time.perf_counter()
        execute_values(cursor, query, chunk)
        conn.commit()
        t1 = time.perf_counter()
        
        chunk_times.append(t1 - t0)
    
    # 5.Cálculo de métricas
    min_time = min(chunk_times)
    max_time = max(chunk_times)
    avg_time = statistics.mean(chunk_times)
    std_time = statistics.stdev(chunk_times) if len(chunk_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Texto [P0] ({len(chunk_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_p0()
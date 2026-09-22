import time
import statistics
from sentence_transformers import SentenceTransformer
from config import load_config
from connect import connect

def create_distance_functions(cursor):
    """Crea funciones PL/SQL para calcular distancias vectoriales sobre arrays FLOAT8[]."""
    cursor.execute("""
        CREATE OR REPLACE FUNCTION euclidean_distance(a FLOAT8[], b FLOAT8[]) 
        RETURNS FLOAT8 AS $$
            SELECT SQRT(SUM((x - y) * (x - y)))
            FROM unnest(a, b) AS t(x, y);
        $$ LANGUAGE SQL IMMUTABLE STRICT;
    """)

    cursor.execute("""
        CREATE OR REPLACE FUNCTION cosine_similarity(a FLOAT8[], b FLOAT8[]) 
        RETURNS FLOAT8 AS $$
            SELECT SUM(x * y) / (SQRT(SUM(x * x)) * SQRT(SUM(y * y)))
            FROM unnest(a, b) AS t(x, y);
        $$ LANGUAGE SQL IMMUTABLE STRICT;
    """)

def print_stats(metric_name, times):
    min_t = min(times)
    max_t = max(times)
    avg_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    
    print(f"\n--- Métricas Búsqueda Top-2 [{metric_name}] (10 oraciones) ---")
    print(f"Mínimo:         {min_t:.6f} s")
    print(f"Máximo:         {max_t:.6f} s")
    print(f"Promedio:       {avg_t:.6f} s")
    print(f"Desv. Estándar: {std_t:.6f} s")

    
def run_p2():
    print("=== [P2] Benchmark de Consultas Vectoriales en PostgreSQL ===")
    
    # 1. Cargar oraciones desde sentences.txt y tomar 10 para la prueba
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    query_sentences = sentences[:10]
    print("Cargadas 10 oraciones de prueba para el benchmark.")
    for text in query_sentences:
            print(f"    Oración: {text}") 
    # 2. Cargar modelo y vectorizar las consultas
    print("Cargando el modelo 'all-MiniLM-L6-v2' para codificar las consultas...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embeddings = model.encode(query_sentences, convert_to_numpy=True)

    # 3. Conectar a PostgreSQL usando config.py y connect.py
    config = load_config()
    conn = connect(config)
    
    if conn is None:
        print("Error: No se pudo establecer la conexión a PostgreSQL.")
        return

    cursor = conn.cursor()

    # 4. Registrar las funciones SQL de cálculo vectorial
    create_distance_functions(cursor)
    conn.commit()

    # 5. Benchmark 1: Distancia Euclídea (L2)
    print("\n--- 1. Búsqueda Top-2 por Distancia Euclídea (L2) ---")

    l2_times = []
    for q_emb in query_embeddings:
        q_vec = q_emb.tolist()
        t0 = time.perf_counter()
        cursor.execute("""
            SELECT text, euclidean_distance(embedding, %s) AS dist
            FROM p1_sentences_vectors
            ORDER BY dist ASC
            LIMIT 2;
        """, (q_vec,))
        results = cursor.fetchall()
        t1 = time.perf_counter()
        
        l2_times.append(t1 - t0)

    print("Resultados de la primera consulta (L2):")
    for text, dist in results:
        print(f"Oración: {text}, Distancia L2: {dist:.4f}") 

    # 6. Benchmark 2: Similitud Coseno
    print("\n--- 2. Búsqueda Top-2 por Similitud Coseno ---")
    cos_times = []
    
    for q_emb in query_embeddings:
        q_vec = q_emb.tolist()
        t0 = time.perf_counter()
        cursor.execute("""
            SELECT text, cosine_similarity(embedding, %s) AS sim
            FROM p1_sentences_vectors
            ORDER BY sim DESC
            LIMIT 2;
        """, (q_vec,))
        results = cursor.fetchall()
        t1 = time.perf_counter()
        
        cos_times.append(t1 - t0)

    print("Resultados de la primera consulta (Coseno):")
    for text, sim in results:
        print(f"Oración: {text}, Similitud Coseno: {sim:.4f}")

    # 7. Imprimir reportes
    print_stats("Distancia Euclídea (L2)", l2_times)
    print_stats("Similitud Coseno", cos_times)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_p2()
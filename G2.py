import time
import statistics
from sentence_transformers import SentenceTransformer
from config import load_config
from connect import connect

def print_stats(metric_name, times):
    min_t = min(times)
    max_t = max(times)
    avg_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    
    print(f"\n--- Métricas Búsqueda Top-2 pgvector [{metric_name}] (10 oraciones) ---")
    print(f"Mínimo:         {min_t:.6f} s")
    print(f"Máximo:         {max_t:.6f} s")
    print(f"Promedio:       {avg_t:.6f} s")
    print(f"Desv. Estándar: {std_t:.6f} s")

def run_g2():
    print("=== [G2] Benchmark de Consultas Vectoriales con pgvector ===")
    
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    query_sentences = sentences[:10]

    # Imprimir las 10 oraciones elegidas
    print("\n--- 10 Oraciones Seleccionadas para la Búsqueda ---")
    for idx, s in enumerate(query_sentences, 1):
        print(f"  {idx}. {s}")

    print("\nCodificando las 10 oraciones con 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embeddings = model.encode(query_sentences, convert_to_numpy=True)

    config = load_config()
    conn = connect(config)
    if conn is None:
        return

    cursor = conn.cursor()

    # 1. Distancia Euclídea (<->)
    l2_times = []
    last_l2_results = []

    for i, q_emb in enumerate(query_embeddings):
        q_vec_str = str(q_emb.tolist())
        
        t0 = time.perf_counter()
        cursor.execute("""
            SELECT text, embedding <-> %s::vector AS dist
            FROM g1_sentences_vectors
            ORDER BY dist ASC
            LIMIT 2;
        """, (q_vec_str,))
        rows = cursor.fetchall()
        t1 = time.perf_counter()
        
        l2_times.append(t1 - t0)
        
        if i == len(query_embeddings) - 1:
            last_l2_results = rows

    # Mostrar matches de la última frase para L2
    print(f"\n--- Resultados Top-2 para la última frase (L2 - pgvector) ---")
    print(f"Oración Consulta: \"{query_sentences[-1]}\"")
    for doc, dist in last_l2_results:
        print(f"  -> Coincidencia: \"{doc}\" | Distancia L2: {dist:.4f}")

    # 2. Distancia Coseno (<=>)
    cos_times = []
    last_cos_results = []

    for i, q_emb in enumerate(query_embeddings):
        q_vec_str = str(q_emb.tolist())
        
        t0 = time.perf_counter()
        cursor.execute("""
            SELECT text, embedding <=> %s::vector AS dist
            FROM g1_sentences_vectors
            ORDER BY dist ASC
            LIMIT 2;
        """, (q_vec_str,))
        rows = cursor.fetchall()
        t1 = time.perf_counter()
        
        cos_times.append(t1 - t0)
        
        if i == len(query_embeddings) - 1:
            last_cos_results = rows

    # Mostrar matches de la última frase para Coseno
    print(f"\n--- Resultados Top-2 para la última frase (Coseno - pgvector) ---")
    print(f"Oración Consulta: \"{query_sentences[-1]}\"")
    for doc, dist in last_cos_results:
        print(f"  -> Coincidencia: \"{doc}\" | Distancia Coseno: {dist:.4f}")

    # Estadísticas globales
    print_stats("Distancia Euclídea (<->)", l2_times)
    print_stats("Distancia Coseno (<=>)", cos_times)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_g2()
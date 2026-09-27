import time
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000

def print_stats(metric_name, times):
    min_t = min(times)
    max_t = max(times)
    avg_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    
    print(f"\n--- Métricas Búsqueda Top-2 ChromaDB [{metric_name}] (10 oraciones) ---")
    print(f"Mínimo:         {min_t:.6f} s")
    print(f"Máximo:         {max_t:.6f} s")
    print(f"Promedio:       {avg_t:.6f} s")
    print(f"Desv. Estándar: {std_t:.6f} s")

def setup_collection_for_metric(client, metric_space, sentences, embeddings):
    """Crea e indexa una colección en ChromaDB configurada para una métrica HNSW específica."""
    coll_name = f"c2_sentences_{metric_space}"
    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": metric_space}  # "l2" o "cosine"
    )
    
    # Inserción en lotes de CHUNK_SIZE
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk_docs = sentences[i : i + CHUNK_SIZE]
        chunk_embs = embeddings[i : i + CHUNK_SIZE].tolist()
        chunk_ids = [f"id_{j}" for j in range(i, i + len(chunk_docs))]
        
        collection.add(
            embeddings=chunk_embs,
            documents=chunk_docs,
            ids=chunk_ids
        )
        
    return collection

def run_c2():
    print("=== [C2] Benchmark de Consultas Vectoriales en ChromaDB ===")
    
    # 1. Cargar oraciones desde sentences.txt y tomar 10 de prueba
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    query_sentences = sentences[:10]
    print(f"Cargadas {len(sentences)} oraciones del corpus.")
    
    # Imprimir las 10 oraciones seleccionadas para la consulta
    print("\n--- 10 Oraciones Seleccionadas para la Búsqueda ---")
    for idx, s in enumerate(query_sentences, 1):
        print(f"  {idx}. {s}")

    # 2. Generar embeddings para todo el corpus y para las 10 consultas
    print("\nCargando el modelo 'all-MiniLM-L6-v2' y calculando embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    all_embeddings = model.encode(sentences, convert_to_numpy=True)
    query_embeddings = all_embeddings[:10].tolist()

    # 3. Inicializar cliente persistente en local
    client = chromadb.PersistentClient(path="./chroma_db")

    # 4. Benchmark 1: Distancia Euclídea (L2)
    print("\n[1/2] Creando e indexando colección HNSW con métrica L2...")
    coll_l2 = setup_collection_for_metric(client, "l2", sentences, all_embeddings)
    
    l2_times = []
    last_l2_results = None
    
    for i, q_emb in enumerate(query_embeddings):
        t0 = time.perf_counter()
        res = coll_l2.query(
            query_embeddings=[q_emb],
            n_results=2
        )
        t1 = time.perf_counter()
        l2_times.append(t1 - t0)
        
        # Guardar el resultado de la última frase (índice 9)
        if i == len(query_embeddings) - 1:
            last_l2_results = res

    # Imprimir los resultados de la última frase para L2
    print(f"\n--- Resultados Top-2 para la última frase (L2) ---")
    print(f"Oración Consulta: \"{query_sentences[-1]}\"")
    if last_l2_results and "documents" in last_l2_results:
        for doc, dist in zip(last_l2_results["documents"][0], last_l2_results["distances"][0]):
            print(f"  -> Coincidencia: \"{doc}\" | Distancia L2: {dist:.4f}")

    # 5. Benchmark 2: Similitud Coseno
    print("\n[2/2] Creando e indexando colección HNSW con métrica Coseno...")
    coll_cos = setup_collection_for_metric(client, "cosine", sentences, all_embeddings)
    
    cos_times = []
    last_cos_results = None
    
    for i, q_emb in enumerate(query_embeddings):
        t0 = time.perf_counter()
        res = coll_cos.query(
            query_embeddings=[q_emb],
            n_results=2
        )
        t1 = time.perf_counter()
        cos_times.append(t1 - t0)
        
        # Guardar el resultado de la última frase (índice 9)
        if i == len(query_embeddings) - 1:
            last_cos_results = res

    # Imprimir los resultados de la última frase para Coseno
    print(f"\n--- Resultados Top-2 para la última frase (Coseno) ---")
    print(f"Oración Consulta: \"{query_sentences[-1]}\"")
    if last_cos_results and "distances" in last_cos_results:
        for doc, dist in zip(last_cos_results["documents"][0], last_cos_results["distances"][0]):
            print(f"  -> Coincidencia: \"{doc}\" | Distancia Coseno: {dist:.4f}")

    # 6. Reporte de métricas estadísticas
    print_stats("Distancia Euclídea (L2)", l2_times)
    print_stats("Similitud Coseno", cos_times)

if __name__ == "__main__":
    run_c2()
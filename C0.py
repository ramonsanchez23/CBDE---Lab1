import time
import statistics
import chromadb

CHUNK_SIZE = 1000

def run_c0():
    print("=== [C0] Carga de Texto Plano en ChromaDB ===")
    
    # 1. Cargar oraciones desde sentences.txt
    with open("sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Cargadas {len(sentences)} oraciones desde 'sentences.txt'.")

    # 2. Inicializar cliente persistente de Chroma
    client = chromadb.PersistentClient(path="./chroma_db")

    # borrar colección existente si existe
    try:
        client.delete_collection(name="c0_sentences")
    except Exception:
        pass

    collection = client.create_collection(name="c0_sentences")

    # 3. Inserción por lotes midiendo cada chunk
    chunk_times = []
    
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk_docs = sentences[i : i + CHUNK_SIZE]
        chunk_ids = [f"doc_{j}" for j in range(i, i + len(chunk_docs))]
        
        t0 = time.perf_counter()
        collection.add(
            documents=chunk_docs,
            ids=chunk_ids
        )
        t1 = time.perf_counter()
        
        chunk_times.append(t1 - t0)

    # 4. Cálculo de métricas estadísticas de almacenamiento de texto
    min_time = min(chunk_times)
    max_time = max(chunk_times)
    avg_time = statistics.mean(chunk_times)
    std_time = statistics.stdev(chunk_times) if len(chunk_times) > 1 else 0.0

    print(f"\n--- Métricas Almacenamiento Texto [C0] ({len(chunk_times)} chunks de tamaño {CHUNK_SIZE}) ---")
    print(f"Mínimo:         {min_time:.6f} s")
    print(f"Máximo:         {max_time:.6f} s")
    print(f"Promedio:       {avg_time:.6f} s")
    print(f"Desv. Estándar: {std_time:.6f} s")

if __name__ == "__main__":
    run_c0()
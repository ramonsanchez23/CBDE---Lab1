from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("1. Descargando subconjunto de bookCorpus desde HuggingFace...")
    dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)
    
    sentences = []
    target_count = 10000
    
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 15:
            sentences.append(text)
        if len(sentences) >= target_count:
            break
            
    print(f"Se recolectaron {len(sentences)} oraciones.")
    
    # Guardar en un archivo .txt plano (una oración por línea)
    output_file = "sentences.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")
            
    print(f"Oraciones guardadas correctamente en '{output_file}'.")

    # Verificar que el modelo de embeddings funciona correctamente
    print("\n2. Verificando descarga del modelo de embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sample_emb = model.encode(sentences[0])
    print(f"Prueba exitosa. Dimensión del vector generado: {len(sample_emb)}")

if __name__ == "__main__":
    main()
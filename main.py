from document_loader import load_documents
from splitter import split_document
from embedding_store import EmbeddingStore
from llm_client import generate_response

# Prompt Builder Function
def build_prompt(context, user_query):
    return f"""
You are an expert AI assistant. Use the following context to answer the user's question. 
If the context is not helpful, just say 'I don't know.' 

Context:
{context}

Question: {user_query}

Answer:"""

# Step 1: Load Documents
docs = load_documents("documents")
store = EmbeddingStore()

# Step 2: Split Documents and Add Embeddings
for doc in docs:
    print(f"📄 File: {doc['metadata']['filename']}")
    chunks = split_document(doc["content"], chunk_size=300, chunk_overlap=50)
    if chunks:
        metadata_list = [{"filename": doc["metadata"]["filename"], "chunk": c} for c in chunks]
        store.add_embeddings(chunks, metadata_list)
        print(f"📝 Number of Chunks: {len(chunks)}")
        print(f"📌 First Chunk Preview:\n{chunks[0][:300]}\n")

store.save()  # Save the FAISS vector store

# Step 3: Perform Retrieval Using MMR
results = store.search_mmr("Explain Artificial Intelligence", k=2)
print("🔎 MMR Search Results:")
for res in results:
    print(f"File: {res['filename']}, Chunk Preview: {res['chunk'][:200]}...\n")

# Step 4: Generate LLM Response
context = "\n\n".join([res['chunk'] for res in results])
prompt = build_prompt(context, "Explain Artificial Intelligence")
answer = generate_response(prompt)

print("🤖 Final Answer:\n", answer)

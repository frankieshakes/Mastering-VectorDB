import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Setup ---
# 1. Define storage and embedding model
CHROMA_PATH = "chroma_data/"
# Use an open-source, powerful embedding model suitable for local development.
# 'all-MiniLM-L6-v2' is fast and effective.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create sample documents with rich metadata
documents = [
    Document(
        page_content="Our Q4 earnings report shows a 15% increase in cloud services revenue, reaching $1.2B.",
        metadata={"source": "Finance_Report", "year": 2024, "access_level": "Executive"}
    ),
    Document(
        page_content="The new microservice architecture dramatically improves latency by 30%, which is documented in the technical whitepaper.",
        metadata={"source": "Technical_Docs", "year": 2024, "access_level": "Developer"}
    ),
    Document(
        page_content="The marketing strategy for Q1 2025 will focus heavily on social media campaigns targeting Gen Z.",
        metadata={"source": "Marketing_Plan", "year": 2025, "access_level": "Public"}
    ),
    Document(
        page_content="The finance team requires all budget requests for Q1 2025 to be submitted by the end of November 2024.",
        metadata={"source": "Internal_Memo", "year": 2024, "access_level": "Executive"}
    ),
]

# --- Ingestion (Creating the Vector Store) ---
print("1. Indexing documents with metadata...")
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)
print(f"   Indexed {len(documents)} documents into Chroma.")
# Persist the data to disk (Chroma saves it locally)
vector_db.persist()

# --- Retrieval (Performing the Search) ---

# Scenario A: Semantic Search ONLY (No Filter)
query_a = "What should we concentrate on for advertising next year?"
results_a = vector_db.similarity_search(query_a, k=2)

print("\n--- Scenario A: Semantic Search (Unfiltered) ---")
print(f"Query: '{query_a}'")
for i, doc in enumerate(results_a):
    print(f"Result {i+1} (Source: {doc.metadata['source']}): {doc.page_content[:60]}...")

# Scenario B: Semantic Search with Metadata Filter
# We only want information from a source with 'Executive' access
# This is crucial for securing data access in RAG systems.
query_b = "What were the financial results from the last period?"
filter_b = {"access_level": "Executive"}
results_b = vector_db.similarity_search(query_b, k=1, filter=filter_b)

print("\n--- Scenario B: Semantic Search with FILTER ('access_level': 'Executive') ---")
print(f"Query: '{query_b}'")

if results_b:
    doc = results_b[0]
    print(f"Result 1 (Source: {doc.metadata['source']}, Year: {doc.metadata['year']}): {doc.page_content[:60]}...")
else:
    print("No documents found matching the filter criteria.")

# Scenario C: Filtered Search with Logical Operator
# Find documents related to 2024 OR having Developer access
query_c = "Tell me about the recent technology updates."
filter_c = {
    "$or": [
        {"year": 2024},
        {"access_level": "Developer"}
    ]
}
results_c = vector_db.similarity_search(query_c, k=3, filter=filter_c)

print("\n--- Scenario C: Filtered Search with OR Operator ---")
print(f"Query: '{query_c}'")
for i, doc in enumerate(results_c):
    print(f"Result {i+1} (Source: {doc.metadata['source']}, Year: {doc.metadata['year']}): {doc.page_content[:60]}...")

# Clean up the persisted directory (optional)
# import shutil
# if os.path.exists(CHROMA_PATH):
#     shutil.rmtree(CHROMA_PATH)
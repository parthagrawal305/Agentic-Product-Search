import os
import uuid
import random
from dotenv import load_dotenv
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

# 1. Initialize Clients
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Use persistent local storage for Vector DB
qdrant = QdrantClient(path="./qdrant_data")

# 2. Recreate Qdrant Collection (Dimension 384 for fastembed)
COLLECTION_NAME = "ecommerce_products"
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
# Fastembed handles embedding text directly!
qdrant.set_model("BAAI/bge-small-en-v1.5")


# 3. Generate Synthetic Indian E-commerce Data
categories = ["Winter Jacket", "Kurta Set", "Silk Saree", "Running Shoes", "Cotton T-Shirt", "Denim Jeans", "Formal Blazer"]
colors = ["Red", "Navy Blue", "Black", "Olive Green", "Mustard Yellow", "White", "Maroon"]
sizes = ["S", "M", "L", "XL", "Free Size", "UK 8", "UK 9"]

products = []

for i in range(150):
    cat = random.choice(categories)
    color = random.choice(colors)
    size = random.choice(sizes)
    
    # Generate realistic INR pricing based on category
    if "Jacket" in cat or "Blazer" in cat or "Saree" in cat:
        price = random.randint(1500, 5000)
    elif "Shoes" in cat:
        price = random.randint(2000, 4000)
    else:
        price = random.randint(400, 1200)
        
    # Ensure test case jacket exists
    if i == 0:
        cat = "Winter Jacket"
        color = "Black"
        price = 99
        title = "Ultra-Light Black Winter Jacket (Clearance)"
    else:
        title = f"{color} {cat}"

    prod = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": f"Premium quality {color.lower()} {cat.lower()}. Perfect for everyday wear or special occasions. Available in size {size}.",
        "price_inr": price,
        "category": cat,
        "color": color,
        "size": size,
        "stock_quantity": random.randint(0, 50)
    }
    products.append(prod)

print(f"Generated {len(products)} products. Seeding databases...")

# 4. Insert into Supabase and Qdrant
# Delete existing supabase rows to prevent duplicate runs
supabase.table("products").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

supabase_batch = []
docs = []
metadata = []
ids = []

for prod in products:
    # Prepare Supabase insert
    supabase_batch.append(prod)
    
    # Prepare Qdrant Vector embedding + Payload
    docs.append(f"{prod['title']} - {prod['description']}")
    metadata.append({
        "id": prod["id"],
        "title": prod["title"],
        "price_inr": prod["price_inr"],
        "category": prod["category"]
    })
    ids.append(prod["id"])
    
print("Uploading to Supabase...")
# Supabase limits bulk inserts, but 150 is fine.
supabase.table("products").insert(supabase_batch).execute()

print("Embedding documents with FastEmbed...")
from fastembed import TextEmbedding
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(embedding_model.embed(docs))

print("Uploading to Qdrant...")
points = [
    PointStruct(
        id=ids[i],
        vector=embeddings[i].tolist(),
        payload=metadata[i]
    )
    for i in range(len(docs))
]
qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ Data Seeding Complete!")

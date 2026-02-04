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
categories = [
    ("Winter Jacket", "clothing", True),
    ("Denim Jeans", "clothing", True),
    ("Running Shoes", "footwear", True),
    ("Cotton T-Shirt", "clothing", True),
    ("Formal Blazer", "clothing", True),
    ("Sun Hat", "accessories", True),
    ("Travel Backpack", "accessories", False),
    ("Graphite Pencil Set", "stationery", False),
    ("Wireless Earbuds", "electronics", False),
    ("Sunglasses", "accessories", True)
]

colors = ["Red", "Navy Blue", "Black", "Olive Green", "Mustard Yellow", "White", "Maroon", "Grey", "Cyan", "Pink"]
sizes_clothing = ["S", "M", "L", "XL"]
sizes_shoes = ["UK 7", "UK 8", "UK 9", "UK 10", "UK 6"]

products = []

for cat_name, cat_type, split_gender in categories:
    for i in range(10):  # 10 products per category
        color = random.choice(colors)
        
        # Gender assignment
        if split_gender:
            gender = "Men's" if i < 5 else "Women's"
        else:
            gender = "Unisex"
            
        # Size assignment
        if cat_type == "clothing":
            size = random.choice(sizes_clothing)
        elif cat_type == "footwear":
            size = random.choice(sizes_shoes)
        else:
            size = "Standard Size"
            
        # Pricing
        if cat_type == "electronics": price = random.randint(1500, 15000)
        elif cat_type == "stationery": price = random.randint(50, 500)
        elif "Jacket" in cat_name or "Blazer" in cat_name: price = random.randint(1500, 5000)
        elif cat_type == "footwear": price = random.randint(2000, 4000)
        else: price = random.randint(400, 1500)
        
        title = f"{gender} {color} {cat_name}"
        desc = f"Premium quality {color.lower()} {cat_name.lower()} for {gender.lower()}. Perfect for everyday use. "
        if size != "Standard Size":
            desc += f"Available in size {size}."
            
        # Add a weight
        weight_g = random.randint(100, 800)
        desc += f" Weight: {weight_g}g."

        prod = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": desc,
            "price_inr": price,
            "category": cat_name,
            "color": color,
            "size": size,
            "stock_quantity": random.randint(0, 50)
        }
        
        # Hardcode test cases
        if len(products) == 0:
            prod["title"] = "Men's Black Winter Jacket (Clearance)"
            prod["price_inr"] = 99
            
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

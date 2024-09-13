import os
import json
import openai
import yaml
#from PIL import Image
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
#from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.vectorstores import Chroma
#from transformers import CLIPProcessor, CLIPModel
from src.db_schema import ListingCollection  # Import the schema

# Load the environment configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract relevant settings from the config file
api_base = config["env_config"]["api_base"]["base1"]
embedding_model = config["env_config"]["model_name"]["text2"]
file_path = config["data"]["listing_file_path"]

api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = api_base
openai.api_key = api_key

# Set up CLIP model and processor for generating multimodal embeddings
#clip_model = CLIPModel.from_pretrained(embedding_model)
#clip_processor = CLIPProcessor.from_pretrained(embedding_model)


# Initialize the custom embedding function
#embedding_function = OpenCLIPEmbeddings()
embedding_function = OpenAIEmbeddings(model=embedding_model)

persist_directory = config["data"]["vector_db_path"]

# Load the listings from the JSON file
def load_listings():
    with open(file_path, 'r') as f:
        listings = json.load(f)
    return listings

'''
# Function to generate multimodal embeddings using CLIP
def generate_clip_embeddings(text, image_path):
    if image_path is None:
        # Use the same dimension as the text embeddings
        return np.zeros(embedding.shape)  # Ensure this dimension matches `embedding.shape`

    # Check if the image path is valid and file exists
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}")

    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    except Exception as e:
        raise RuntimeError(f"Failed to open image at {image_path}: {e}")

    # Process the text and image
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    # Get the image features
    image_features = clip_model.get_image_features(**inputs)
    
    return image_features.detach().numpy()
'''

def store_listings_in_vector_db():
    global vectorstore

    # Remove existing data by deleting the persist directory
    #if os.path.exists(persist_directory):
     #   shutil.rmtree(persist_directory)

    # Initialize the vectorstore
    vectorstore = Chroma(
        collection_name="real_estate_listings",
        persist_directory=persist_directory, 
        embedding_function=embedding_function)
    
    # Clear existing data (if any)
    #vectorstore._collection.delete_many({})
    
    # Load listings from file
    listing_collection_dict = load_listings()

    # Parse the listings
    listings = ListingCollection(listings=listing_collection_dict).listings

    # Process each listing and store in ChromaDB
    for idx, listing in enumerate(listings):
        try:
            # Create the listing text to embed
            listing_text = f"{listing.neighborhood} | {listing.price} | {listing.bedrooms} bedrooms | " \
                       f"{listing.bathrooms} bathrooms | {listing.square_footage} sqft | " \
                       f"Description: {listing.description} | " \
                       f"Neighborhood Description: {listing.neighborhood_description}"

            # Generate embeddings for the listing text
            embedding = embedding_function.embed_documents([listing_text])[0]
            embedding = np.array(embedding)  # Convert to NumPy array

            '''
            # Generate image embedding using CLIP
            if listing.image_path:
                image_embedding = generate_clip_embeddings(listing_text, listing.image_path)
            else:
                # Use default vector with the same shape as `embedding`
                image_embedding = np.zeros(embedding.shape)

            # Combine text and image embeddings into a single vector
            if embedding.shape != image_embedding.shape:
                raise ValueError(f"Shape mismatch: embedding shape {embedding.shape}, image embedding shape {image_embedding.shape}")
        
            combined_embedding = embedding + image_embedding.flatten()
            '''
            combined_embedding = embedding

            # Add the listing and its embedding to ChromaDB
            vectorstore.add_texts([listing_text], embedding_vectors=[combined_embedding])
            #print(f"Added listing {idx+1}/{len(listings)} to vectorstore.")

        except Exception as e:
            print(f"Error processing listing {idx+1}/{len(listings)}: {e}")

    # Persist the ChromaDB data to disk
    vectorstore.persist()
    print("Listings have been stored in the vector database.")

    # Debugging: Check the number of documents in the vectorstore
    num_documents = vectorstore._collection.count()
    print(f"Number of documents in the vectorstore: {num_documents}")

def main():
    store_listings_in_vector_db()

if __name__ == "__main__":
    main()
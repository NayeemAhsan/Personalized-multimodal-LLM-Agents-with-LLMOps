import os
import json
import yaml
import torch
import openai
import pydantic
from diffusers import DiffusionPipeline
from src.db_schema import RealEstateListing 

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract relevant settings from the config file
api_base = config["env_config"]["api_base"]["base1"]
temperature = config['data']['temperature']

api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = api_base
openai.api_key = api_key

def generate_listing(listing_list_range):
    model_name = config["env_config"]["model_name"]["text1"]
    listings = []


    for _ in range(listing_list_range):
        response = openai.ChatCompletion.create(
            model=model_name,
            temperature = temperature,
            messages=[
                {"role": config['agents']['system_text_generator']['role'].strip(), 
                 "content": config["prompts"]["listing_generator"]["description"]}
            ]
        )
        listing_content = response['choices'][0]['message']['content']
        
        listings.append(listing_content)

    return listings

def generate_image(description):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure correct dtype based on device
    if device == "cpu":
        pipe = DiffusionPipeline.from_pretrained(config["env_config"]["model_name"]["image1"])
    else:
        pipe = DiffusionPipeline.from_pretrained(
            config["env_config"]["model_name"]["image1"], 
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to(device)
    
    # Truncate the description to fit within 77 tokens for CLIP
    truncated_description = " ".join(description.split()[:75])
    
    image = pipe(
        prompt=truncated_description, 
        num_inference_steps=50,  
        guidance_scale=7.5,
    ).images[0]
    
    # Save image to a file and return the path
    image_path = f"images/{truncated_description[:50].replace(' ', '_')}.png"
    image.save(image_path)
    
    return image_path

def main():
    listing_list_range = config["data"]["listing_list_range"]
    image_list_range = config["data"]["image_list"]
    
    listings = generate_listing(listing_list_range)
    listings_with_images = []

    for i, listing_text in enumerate(listings):
        try:
            # Parse the JSON string into a Python dictionary
            listing_dict = json.loads(listing_text)
            
            # Initial parsing without `image_path`
            listing_data = RealEstateListing.construct(**listing_dict)
            
            # Generate images only for the first `image_list_range` listings
            if i < image_list_range:
                listing_data.image_path = generate_image(listing_data.description)
            else:
                listing_data.image_path = None
            
            # Re-validate the listing data with the image path included
            listing_data = RealEstateListing.parse_obj(listing_data.dict())

            # Add the listing with image path to the list
            listings_with_images.append(listing_data.dict())

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            continue
        except pydantic.error_wrappers.ValidationError as e:
            print(f"Validation error: {e}")
            continue
        
    # Save the listings and images to a single JSON file
    output_file = config["data"]["listing_file_path"]
    with open(output_file, "w") as f:
        json.dump(listings_with_images, f, indent=4)

    print(f"Generated {listing_list_range} listings and saved to {output_file}")

    # Print the first two listings for verification
    print(f"First two listings:\n{json.dumps(listings_with_images[:2], indent=4)}")


if __name__ == "__main__":
    main()

import os
import openai
import yaml
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import comet_llm
#from langchain_experimental.open_clip import OpenCLIPEmbeddings
#from transformers import CLIPProcessor, CLIPModel

# logging in Comet LLM
comet_llm.init(api_key = os.environ["COMET_API_KEY"],
               project="Personalized_Real_estate_agent"
               #workspace="real_estate_agents_practice"
               )

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract relevant settings from the config file
api_base = config["env_config"]["api_base"]["base1"]
embedding_model = config["env_config"]["model_name"]["text2"]
#file_path = config["data"]["listing_file_path"]

# api_key for openai
api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = api_base
openai.api_key = api_key

# Set up CLIP model and processor for generating multimodal embeddings
#clip_model = CLIPModel.from_pretrained(embedding_model)
#clip_processor = CLIPProcessor.from_pretrained(embedding_model)

# Initialize the OpenAI embedding function
#embedding_function = OpenCLIPEmbeddings()
embedding_function = OpenAIEmbeddings(model=embedding_model)

# Initialize the vectorstore with persistence
persist_directory = config["data"]["vector_db_path"]
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function,
    collection_name="real_estate_listings"
)

# Debugging: Check the number of documents in the vectorstore
print(f"Number of documents in the vectorstore: {vectorstore._collection.count()}")

def get_user_responses():
    user_prompt = config['prompts']['user_prompt_generator']['description']
    response = openai.ChatCompletion.create(
        model=config['env_config']['model_name']['text1'],
        messages=[
            {"role": config['agents']['user_prompt_generator']['role'].strip(), 
             "content": user_prompt}
        ]
    )
    #comet_llm.log_prompt(
     #       prompt=user_prompt,
      #      output=response['choices'][0]['message']['content']
       #     )
    return response['choices'][0]['message']['content']

def search_and_retrieve(query, k=5):
    # Ensure the query is a string (if it's a list, join it into a single string)
    if isinstance(query, list):
        query = " ".join(query)

    # Split the query text using CharacterTextSplitter
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(query)
    
    results = []
    for chunk in chunks:
        chunk_results = vectorstore.similarity_search(chunk, k=k)
        results.extend(chunk_results)
    
    return results

def generate_final_response(user_responses, num_results=1):

    prompt = config['prompts']['user_text_generator']['description']
    num_results = config['data']['top-k']

    final_results = []
    for response in [user_responses]:
        listings = search_and_retrieve(query=response, k=num_results)
        personalized_listings = []

        for listing in listings:
            # Access the metadata directly from the Document object           
            # create the context
            content = prompt.format(preferences=response, 
                                description = listing.metadata.get('description', 'No description available'), 
                                neighborhood_description = listing.metadata.get('neighborhood_description', 'No neighborhood description available'),
                                price = listing.metadata.get('price', 'No price available'))
            
            final_response = openai.ChatCompletion.create(
                model=config['env_config']['model_name']['text1'],
                messages=[
                    {"role": config['agents']['user_text_generator']['role'].strip(), 
                     "content": content}
                ]
            )
            personalized_listings.append({
                'personalized_description': final_response['choices'][0]['message']['content']
            })

        final_results.append({
            'user_response': response,
            'personalized_listings': personalized_listings
        })

    # Store the personalized results in a JSON file
    output_file_path = config["data"]["results_file_path"]
    with open(output_file_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Personalized results saved to {output_file_path}")

def main(user_responses, num_results=1):
    generate_final_response(user_responses, num_results)

    # Load the JSON file and extract 'personalized_description'
    with open(config["data"]["results_file_path"], 'r') as f:
        personalized_results = json.load(f)
    
    # Extract the personalized listings
    personalized_results[0]['personalized_listings'] = personalized_results[0]['personalized_listings'][:]

    for result in personalized_results:
        for listing in result['personalized_listings']:
            listing['personalized_description']
    

    # start chain
    comet_llm.start_chain(
        inputs={"system's_input": config['prompts']['user_prompt_generator']['description']},
        )

    # store full response
    with comet_llm.Span(
        category="system's_input",
        name="system's_input",
        inputs={"system's_input": config['prompts']['user_prompt_generator']['description']},
        ) as span:
        span.set_outputs(outputs={"system_generated_prompt": user_responses})

    # end chain with the end response after processing it
    with comet_llm.Span(
        category="system_generated_prompt",
        name="system_generated_prompt/user_response",
        inputs={"system_generated_prompt": user_responses,},
        ) as span:

        span.set_outputs(outputs={"personalized_response": listing['personalized_description']})

    comet_llm.end_chain(
        outputs={"personalized_response": listing['personalized_description']},
        metadata = {
                    "model": config['env_config']['model_name']['text1'],
                    "top-k": num_results,
                    "temperature": config['data']['temperature'],
                    }
                    )

if __name__ == "__main__":
    user_responses = [get_user_responses()]  # Ensure it's a list
    main(user_responses)

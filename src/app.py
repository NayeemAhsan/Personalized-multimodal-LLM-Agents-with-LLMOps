import os
import json
import yaml
import gradio as gr
import src.vector_db as vector_db
import src.personalized_response as personalized_response

# Load the environment configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract relevant settings from the config file
results_file_path = config["data"]["results_file_path"]

# Check if vector database is populated
def check_vector_db():
    vector_db.main()  
    if vector_db.vectorstore._collection.count() == 0:
        print("No data in the vector database. Please add listings.")
        return False
    print(f"Number of documents in the vectorstore: {vector_db.vectorstore._collection.count()}")
    return True

# Generate buyer preference
def generate_buyer_pref():
    return personalized_response.get_user_responses()

# Submit buyer preference and number of listings
def submit_pref(buyer_pref, num_listings):
    # Ensure the buyer preference is not empty
    if not buyer_pref:
        return "Error: Buyer preference is empty. Please generate a preference first."

    # Ensure the number of listings is valid
    #if num_listings < 1 or num_listings > 5:
     #   return "Error: Number of listings should be between 1 and 5."

    # Run the main function in personalized_response with user inputs
    personalized_response.main(user_responses=[buyer_pref], num_results=num_listings)

    # Load the JSON file and extract 'personalized_description'
    try:
        with open(results_file_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        return f"Error loading JSON file: {str(e)}"

    # Extract the personalized descriptions for all requested listings
    descriptions = []
    personalized_listings = results[0].get('personalized_listings', [])

    for i in range(min(num_listings, len(personalized_listings))):
        description = personalized_listings[i].get('personalized_description', 'No description available')
        # Format description with bullet points
        formatted_description = "<ul><li>" + description.replace("\n", "</li><li>") + "</li></ul>"
        descriptions.append(formatted_description)

    # Combine the descriptions into a single HTML string
    return "".join(descriptions)


def main():
    # Check if vector database is populated
    #db_status = check_vector_db()
    #if not db_status:
    #    return  # Exit the app if no data in vector database

    with gr.Blocks() as app:
        # Display vector database status
        #gr.Markdown(f"## Vector Database Status: {'Populated' if db_status else 'Not Populated'}")

        with gr.Row():
            buyer_pref_textbox = gr.Textbox(value="", label="Buyer Preference", 
                                            placeholder="Select Generate Buyer Preference", 
                                            lines=3, show_label=True)
            pref_button = gr.Button("Generate Buyer Preference")
            num = gr.Slider(minimum=1, maximum=5, step=1, label="Number of Listings")
            submit_button = gr.Button("Show Matching Listings")
            personalized_descriptions = gr.HTML(label="Personalized Descriptions")

        pref_button.click(generate_buyer_pref, outputs=buyer_pref_textbox)
        submit_button.click(submit_pref, inputs=[buyer_pref_textbox, num], outputs=personalized_descriptions)

        close_button = gr.Button('Close App')
        close_button.click(lambda: app.close())  

        app.launch()

if __name__ == "__main__":
    main()

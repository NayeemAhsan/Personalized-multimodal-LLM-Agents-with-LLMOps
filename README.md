# Personalized-multimodal-LLM-Agents-with-LLMOps

The purpose of this project is to create a personalized multimodal LLM agents and use LLMOps tools like `comet-llm` to track and monitor prompts. I have implemented an real estate agent that tries to personalize experience for each buyer of a real estate company, making the property search process more engaging and tailored to individual preferences. To achieve this, an application called "RealEstateMatch" was developed, which uses Large Language Models (LLMs) and vector databases to transform standard real estate listings into personalized narratives that reflect with potential buyers' unique preferences and needs. 

## Key Features of "RealEstateMatch"

1. Geenrate a synthetic listings by understanding Buyer Preferences using Prompt Engineering

2. Integrate the list with a Vector Database

3. Create a list of personalized description real estate based on similarity search from the vector db

4. Present the list on a prototype app provided by Gradio

5. Utilize `commet-llm` to track and monitor prompts

## Configuration and Dependencies
- All required dependencies are mentioned in the `environment.yml` file.
- All variables like prompts, prompt roles, file locations, temprature settings, top-k, models, embeddings, etc. are located at the `config.yaml` file. Since variables are not hardcoded, it's recommended to make changes to the variables only from this file.
- It's better to have `PYTHONPATH` variable setup at the home directory when running this project. 
- It's recommended to setup OPENAI_API_KEY at the os level and call it using `os.environ["OPENAI_API_KEY"]` from the code. 
- Similarly, setup commet_API_KEY at the os level and call it using `os.environ["COMMET_API_KEY"]` from the code.
- Use the commet portal to track and monitor prompts.

## Execution

#### Genereate a list of real estate
To generate a list of real estate listings, run
`python src/generate_listings.py`

#### Store in a vector db
- DB schema will be found at the db_schema.py file
- To store in a chroma db, run `python src/vector_db.py`

#### Generate Personalized Results
To generate a list of personalized results, run
`python src/personalized_response.py`

#### Gradio app
To open the app, run
`python src/app.py`

#### Jupyter notebook
The project can be run from the `RealEstateMatch.ipynb` located at the home directory by following the directions.

#### Disclaimer
Due to lack of GPU resources and time, the example from this project does not have any images generated. But the code is able to generate images as well as it should be able to do multimodal searches. Feel free to test them! 


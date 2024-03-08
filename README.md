# RAGs-to-riches
Turning your RAGs to riches using LangChain and Streamlit

# Setup
Below are a set of instruction on how to set up the various tools and services required.

## Python virtual environment
All of the following commands should be run in a terminal on a machine with python installed a python download can be found [here](https://www.python.org/downloads/).
1) Create the virtual environment:
```
py -m venv .venv
```
2) Activate the virtual enviornment:
```
.\.venv\Scripts\activate
```
3) Done. It is as easy as that!
Bonus step is to install of all the required python packages from the requirements.txt
- Install the requirements:
```
pip install -r requirements.txt
```


## Setup Azure OpenAI
All of the GenAI examples in this toolbox use langchain and Azure OpenAI for model hosting. If you do not have azure openAI you can get it here [here](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

Once you have access you will need to deploy two models:
1) GPT-3.5 turbo 16k, calling it `ab-test-gpt35-t`
2) Ada embeddings model, calling it `ab-test-ada-002`


## Configure Streamlit
To allow the Streamlit app to access MLFlow it logs into the databricks api_client using credentials stored in a streamlit secrets file.
1) Create a file called *secrets.toml* in the *.streamlit* folder
2) Add the following to secrets.toml:
```
AZURE_OPENAI_ENDPOINT = "<base>"
AZURE_OPENAI_API_KEY = "<key>"
```


## Run the Streamlit app
1)In a terminal navigate to the root folder of the repo
2) Run the app
```
streamlit run app.py
```
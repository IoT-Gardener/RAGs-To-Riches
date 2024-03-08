import os
import tempfile
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
from PIL import Image


os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]

# Get relative path
img_path = Path(__file__).parents[0]
# Load images
logo_img = Image.open(f"{img_path}/Page_Images/HighResIconWithTextLong.png")
# Set the page title and icon and set layout to "wide" to minimise margins
st.set_page_config(page_title="Talk To Your PDF", page_icon=":bookmark_tabs:")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialise vector db
if "vectordb" not in st.session_state:
    st.session_state.vectordb = []

# Initialise vector llm
if "llm" not in st.session_state:
    st.session_state.llm = AzureChatOpenAI(
                openai_api_version="2023-05-15",
                deployment_name="ab-test-gpt35-t"
            )


def main():
    # Add the AA logo to the top of the page
    st.image(logo_img, use_column_width='auto')
    st.title("Talk to your PDF")
    st.write('This demo takes a PDF uploaded by the user, embeds it into a Chroma VectorDB and uses Azure OpenAI '
             'and Langchain to summarize the document and allow the users to interact with a chatbot to interogate it.')
    st.write("---")

    st.markdown(
        """
        ### Steps:
        1. Upload PDF File
        2. Press analyse!
        3. The Model will summarise your document
        4. Ask the model any further questions
        
        **Note : This is using Azure OpenAI, so YOUR data is YOUR Data**
        
        *If you want to talk to a new pdf simply upload a new document and press analyse*
        """
    )
    st.write("---")

    st.header("Summarise your PDF 📄")
    source_doc = st.file_uploader("Upload Source Document", type=["pdf"], accept_multiple_files=False)
    if "do_the_thing" not in st.session_state:
        st.session_state.do_the_thing = False

    # Check if the 'Summarize' button is clicked
    if st.button("Analyse"):
        if source_doc is None:
            st.write("Please upload a document first, then try again")
        else:
            st.session_state.do_the_thing = True
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)
            st.success(f"Loaded {len(pages)} pages from {source_doc.name}.")

            # Create embeddings model
            embeddings = AzureOpenAIEmbeddings(
                            azure_deployment="ab-test-ada-002",
                            openai_api_version="2023-05-15",
                         )

            # Insert the embeddings into a Chroma DB
            st.session_state.vectordb = Chroma.from_documents(pages, embeddings)

            # Create a sumarize chain using the LLM
            chain = load_summarize_chain(st.session_state.llm, chain_type="stuff")

            # Search the vector db
            search = st.session_state.vectordb.similarity_search(" ")

            # Generate a summary of the document
            with st.spinner("Querying model, please wait..."):
                summary = chain.run(input_documents=search, question="""Write a summary of the document in 150 words or less.
                                                                      In the summary describe what type of content it is and
                                                                      overview the content included and any key points.""")

            # Add the model response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": summary})

    if st.session_state.do_the_thing:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask the Document a Question"):
            # Add the users message to the chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                # Display the message
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Search for similar documents in the knowledge base
                docs = st.session_state.vectordb.similarity_search(prompt)
                # Load question answering chain
                chain = load_qa_with_sources_chain(st.session_state.llm, chain_type="stuff")
                # Call the chain using the documents from the vector search and users prompt
                response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                # Display the model response
                st.markdown(response['output_text'])
                # Add the model response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": response['output_text']})


# Run the app
if __name__ == '__main__':
    main()

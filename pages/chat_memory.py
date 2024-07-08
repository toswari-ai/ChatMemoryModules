# Using Langchain library
#
# installation:
#   - pip install clarifai streamlit

# Run: streamlit run app.py
#

import os
import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Clarifai
from langchain.chains import RetrievalQA
from clarifai.modules.css import ClarifaiStreamlitCSS

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Clarifai

# Clarifai Configuration
CLARIFAI_PAT = 'b5d78d6de57f41f7ada7c9f89ce84646'  # Your Personal Access Token (PAT)
USER_ID = 'toswari-ai'
APP_ID = 'teddy-pdf-chat'


#use multi model - Gpt-4o
#model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o"
llm = None
initialized = False

st.set_page_config(page_title="Chat with Documents", page_icon="🦜")
st.title("🦜 Clarifai RAG and Langchain Memory")

ClarifaiStreamlitCSS.insert_default_css(st)

# 1. Data Organization: chunk documents

def load_chunk_pdf(uploaded_files):
    # Read documents
    documents = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def QandA(CLARIFAI_PAT, model_option):
    
    template = """You are a chatbot having a conversation with a human.
        Given the following extracted parts of a long document and a question, create a final answer.
    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    if model_option == "GPT-4" or model_option == "GPT-3_5-turbo" or model_option == "gpt-4o":
        # this is configuration for openai chat-completion model
        USER_ID = "openai"
        APP_ID = "chat-completion"
        MODEL_ID = model_option
    else:
        # this is configuration for mistralai chat-completion model
        USER_ID = "mistralai"
        APP_ID = "completion"
        MODEL_ID = model_option    

    # LLM to use (set to model,user_id,app above)
    from langchain.llms import Clarifai
    clarifai_llm = Clarifai(
        pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=model_option)

    #this is where the magic happens, it combines the LLM with the prompt and memory
    chain = load_qa_chain(
        clarifai_llm, chain_type="stuff", memory=memory, prompt=prompt
    )

    return chain

def main():

    global conversation_initialize
    conversation_initialize = False

    user_question = st.text_input("Ask a question to LLM model about your documents and click on get the response")

    with st.sidebar:
        st.subheader("Please upload your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)

        # Pull-down menu options
        options = ["GPT-4", "GPT-3_5-turbo", "gpt-4o","mistral-7B-Instruct"]
        selected_option = st.selectbox("Choose your llm model option:", options)



    if not (uploaded_files) and selected_option:
        st.info("Please Upload files and Select the LLM model option to continue.")

    elif st.button("Get the response"):
        with st.spinner("Processing"):
            
            # Check if 'conversation_qa' is already in session state, if not, initialize it
            if "conversation_qa" not in st.session_state:
                print("Initializing conversation_qa")
                # Assuming load_chunk_pdf and QandA are defined elsewhere and work as expected
                st.session_state.docs = load_chunk_pdf(uploaded_files)
                st.session_state.conversation_qa = QandA(CLARIFAI_PAT, selected_option)

            # Use the persistent conversation_qa from st.session_state
            response = st.session_state.conversation_qa({"input_documents": st.session_state.docs, "human_input": user_question}, return_only_outputs=True)
            st.success(response['output_text'])
            print(st.session_state.conversation_qa.memory.buffer)

    #elif st.button("Clear Conversation Memory"):
    #    with st.spinner("Processing"):

            # Clear the conversation memory
            #st.session_state.conversation_qa.memory.clear()
            #st.success("Conversation Memory Cleared")
            #print(st.session_state.conversation_qa.memory.buffer)  

if __name__ == '__main__':
    main()
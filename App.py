# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

tokenizer = AutoTokenizer.from_pretrained("Yale-LILY/reastap-large-finetuned-wtq")
model = AutoModelForSeq2SeqLM.from_pretrained("Yale-LILY/reastap-large-finetuned-wtq")

# Set the title for the Streamlit app
st.title("Llama2 Chat CSV - 🦜🦙")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")

# Handle file upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Create a conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function for conversational chat
    def conversational_chat(query):
        encoding = tokenizer(table=table, query=query, return_tensors="pt")
        outputs = model.generate(**encoding)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        st.session_state['history'].append((query, answer))
        return answer

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# Import the necessary libraries
import streamlit as st
from groq import Groq
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

# load apikey details
api_key = st.secrets["GROQ_API_KEY"]

# create an instance of groq
client = Groq(api_key=api_key)

model = joblib.load("chatbot__nlpmodel.pkl")
vectorizer = joblib.load("tfidfvectorizer.pkl")

# Data Ingestion
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

def predict_intent(userip):
    userip = vectorizer.transform([userip]).toarray()
    return model.predict(userip)[0] # predict returns an array, we take the first element

def chabot_response(text):
    intent = predict_intent(text)

    if intent in df['intent']:
        return df[df['intent']==intent]["response"][0]
    else:
        return llama_response

def llama_response(text):
    stream = client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content":"you are a helpful assistant"
            },
            {
                "role":"user",
                "content":text
            }
        ],
        model = "llama-3.3-70b-versatile",
        stream=True
    )
    for chunk in stream:
        response = chunk.choices[0].delta.content
        if response is not None:
            yield response

# start building streamlit ui
st.title("Llama 3.3 Model")
st.subheader("by Sindhura Nadendla")

text = st.text_area("Please ask any question : ")

if text:
    st.subheader("Model Response : ")

    # Get response from the chatbot(either pre-defined or AI generated)
    response = chabot_response(text)

    # Handle both static and streamed responses
    if isinstance(response,str):
        st.write(response) # Predefined Response
    else:
        for chunk in response:
            st.write(chunk) 
        #st.write_stream(response) # Llama response
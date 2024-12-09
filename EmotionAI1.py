import os
import streamlit as st
import torch
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForSequenceClassification
from markupsafe import Markup
import requests
import json
import re
import numpy as np
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load API Key từ biến môi trường
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB1vMQ9bbFJ_zW26cf9-gtaLdpq7AQGVF4")

# Load emotion analysis model
try:
    emotion_model = RobertaForSequenceClassification.from_pretrained(
        r"d:/DAP391m/Assigment final/Assignment_DAP - Copy - Copy/model_emotion/save_weight_1"
    )
except Exception as e:
    st.error(f"Failed to load the emotion model: {e}")

vectorizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

label_mapping = {
    0: "Other",
    1: "Enjoyment🥰",
    2: "Sadness😔",
}

def calculate_emotion_percentages(predictions):
    total_predictions = predictions.sum()
    if total_predictions == 0:
        return np.zeros_like(predictions)
    percentages = (predictions / total_predictions) * 100
    return percentages

def map_position_to_label(position):
    return label_mapping.get(position, "Unknown")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[.,!?])\s+', '', text)
    text = re.sub(r'\s(?=[.,!?])', '', text)
    text = re.sub(r'([.,!?;:])(?=\S)', r'\1 ', text)
    return text.strip()

def get_ChatGoogleGenerativeAI_response(text):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=API_KEY,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in giving advice based on user emotion and text.",
                ),
                ("human", "{advice_prompt}"),
            ]
        )
        chain = prompt | llm
        advice_request = f"My message is: '{user_prompt}', and I'm feeling {result_label}. Can you give me some advice?"
        response = chain.invoke({"advice_prompt": advice_request})
        return response.content
    except Exception as e:
        st.error(f"Failed to generate a response: {e}")
        return "Error"

def predict_emotion(text):
    inputs = vectorizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = emotion_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0].cpu().detach().numpy()

def save_chat_history():
    with open("chat_history.pkl", "wb") as file:
        pickle.dump(st.session_state.messages, file)

def load_chat_history():
    try:
        with open("chat_history.pkl", "rb") as file:
            st.session_state.messages = pickle.load(file)
    except FileNotFoundError:
        st.session_state.messages = []

st.title("Emotion Chatbot")
st.caption("🚀 Emotion Chatbot 🚀")

# Sidebar setup
with st.sidebar:
    st.markdown("# Chat Options")
    mode = st.selectbox("Select Mode:", ["ask_anything", "ask_emotion"])
    if mode == "ask_emotion":
        selected_chart = st.selectbox("Select chart type", ["bar", "pie"])
    if st.button("Save conversation history"):
        save_chat_history()
        st.success("Conversation history saved!")
    if st.button("Reload conversation history"):
        load_chat_history()
        st.success("Conversation history reloaded!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What questions do you have?"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("Generating response..."):
        if mode == "ask_anything":
            response = get_ChatGoogleGenerativeAI_response(user_prompt)

        elif mode == "ask_emotion":
            predictions = predict_emotion(user_prompt)
            result_label = map_position_to_label(np.argmax(predictions))
            percentages = calculate_emotion_percentages(predictions)
            result_label = map_position_to_label(np.argmax(predictions))
            negative_emotions = ["Sadness😔"]
            
            if result_label in negative_emotions:
                advice_prompt = f"My message is: '{user_prompt}', and I'm feeling {result_label}. Can you give me some advice?"                
                advice = get_ChatGoogleGenerativeAI_response(advice_prompt)
                response = f"I sense that you are {result_label}. Consider this advice: {advice}"
            else:
                response = f"Your sentiment is: {result_label}. You seem fine, no advice needed!"

            formatted_percentages = {label: percentage for label, percentage in zip(label_mapping.values(), percentages)}
            #response = f"Emotion detected: {result_label}. Emotion distribution: {formatted_percentages}."

            if selected_chart == "bar":
                bar_chart = px.bar(
                    x=list(formatted_percentages.values()),
                    y=list(formatted_percentages.keys()),
                    labels={"x": "Percentage", "y": "Emotion"},
                    orientation="h",
                )
                st.plotly_chart(bar_chart)
            elif selected_chart == "pie":
                pie_chart = px.pie(
                    names=list(formatted_percentages.keys()),
                    values=list(formatted_percentages.values()),
                )
                st.plotly_chart(pie_chart)
        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages[-1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

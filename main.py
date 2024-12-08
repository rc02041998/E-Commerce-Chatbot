import streamlit as st
import pandas as pd
# from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch
import re

# Load the Flipkart dataset
@st.cache_data
def load_data():
    file_path = "/Users/rohit/Desktop/Assignment/Self_Learning/flipkart_com-ecommerce_sample.csv"
    df = pd.read_csv(file_path)
    # Clean and preprocess the dataset
    df["retail_price"] = df["retail_price"].replace(r"[^\d.]", "", regex=True).astype(float)
    df["context"] = (
    "Product Name: " + df["product_name"] + ". " +
    "Category: " + df["product_category_tree"] + ". " +
    "Price: ₹" + df["retail_price"].astype(str) + ". " +
    "Description: " + df["description"]
    )
    return df

df = load_data()

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    # model = BertForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", load_in_8bit=True)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit App
st.title("E-Commerce Chatbot")
st.sidebar.header("Options")

# Select a product
product_name = st.sidebar.selectbox("Select a Product", df["product_name"].unique())
selected_product = df[df["product_name"] == product_name].iloc[0]
context = selected_product["context"]



# Display selected product details
st.write("### Selected Product Details")
st.write(f"**Name**: {selected_product['product_name']}")
st.write(f"**Category**: {selected_product['product_category_tree']}")
st.write(f"**Price**: ₹{selected_product['retail_price']}")
st.write(f"**Description**: {selected_product['description']}")

# User question input
question = st.text_input("Ask a question about this product:")
prompt_template = f"""
Following are the details of a product:

{context}

Based on the given details answer the following question:

Question: {question}
Answer: 
"""

if st.button("Get Answer"):
    if question:
        # try:
            st.write(prompt_template)
            # Encode the input for the model
            # inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
            # inputs = tokenizer.encode(prompt_template, return_tensors="pt", )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

            # input_ids = inputs#["input_ids"]
            # attention_mask = inputs["attention_mask"]
            
            # Get model outputs
            # outputs = model(input_ids=input_ids)
            outputs = model.generate(input_ids)
            outputs = tokenizer.decode(outputs[0])
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Extract the most probable start and end positions

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            # Decode the full answer
            full_answer = tokenizer.decode(input_ids[0][answer_start:answer_end])

            # Debug: print the full answer to inspect its content
            st.write(f"Full Answer: {full_answer}")

            # Post-processing: Extract price if the question is about price
            if "price" in question.lower():
                # Regex to capture price formats like ₹999, ₹999.99, etc.
                price_match = re.search(r"₹\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", full_answer)
                if price_match:
                    answer = price_match.group(0)
                else:
                    answer = "Price not found."
            else:
                answer = full_answer

            # Display the answer
            st.write(f"### Answer: {answer}")
        # except Exception as e:
        #     st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please enter a question!")

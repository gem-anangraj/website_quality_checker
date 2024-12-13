import streamlit as st
import boto3
import requests
from bs4 import BeautifulSoup
import json
import json_repair
import re

# Set up the Bedrock client
client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")
system_prompt = ("You are an intelligent website quality processing model trained to analyse the website data and "
                 "generate quality score and suggestions for improvement based on multiple criteria.")

# Function to fetch text from a website
def fetch_website_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = ' '.join([element.get_text() for element in soup.find_all(['p', 'span', 'li'])])
        return page_text.strip()
    except requests.exceptions.RequestException as e:
        return f"Error fetching the website content: {e}"


def extract_json_from_string(data_str):
    try:
        pattern = r'\{.*\}'  # Regex to match JSON-like content
        match = re.search(pattern, data_str, re.DOTALL)
        if match:
            return match.group()
        else:
            return None
    except Exception as e:
        return {"error": f"Failed to extract JSON: {e}"}


def bedrock_llama3_1(prompt, model, temperature=0, sys_prompt=system_prompt):
    print(f'model name{model}')
    # Define Prompt Template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    # Define the payload
    payload = {
        "modelId": model,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "prompt": prompt,
            "temperature": temperature,
            "max_gen_len": 2048,
        })
    }

    # call Invoke model
    response = client.invoke_model(
        modelId=payload["modelId"],
        contentType=payload["contentType"],
        accept=payload["accept"],
        body=payload["body"]
    )

    # Print the response
    llm_response = json.loads(response['body'].read())
    print(f'checking model{llm_response}')
    # return llm response
    return llm_response['generation']


# Function to evaluate website text quality
def evaluate_text_quality_bedrock(text):

    if not text:
        return ("The webpage contains no readable text to evaluate.", "", "")
    with open("schema.json", "r") as file:
        json_schema = json.load(file)

    prompt = f"""extract relevant fields from the provided website text.
            input:
                website text: {text}
            output:
                JSON object strictly following the schema: {json_schema}

            instructions:
            - follow the schema exactly.
            - if any field is missing, assign null.
            - use only actual text information; mark fields as null if missing."""


    try:
        result = bedrock_llama3_1(prompt, model="meta.llama3-1-8b-instruct-v1:0")
        parsed_result = extract_json_from_string(result)
        parsed_result = json_repair.loads(parsed_result)
        return parsed_result

    except Exception as e:
        return f"Error evaluating text: {e}"


# Streamlit App
st.title("Hi Website Text Quality Evaluator modified")
st.write("Evaluate the quality of text extracted from a given website using AWS Bedrock. modified")

# Input URL
url = st.text_input("Enter the website URL:")

if st.button("Evaluate"):
    if url:
        with st.spinner("Fetching content from the website..."):
            website_text = fetch_website_text(url)

        # if "Error" in website_text:
        #     st.error(website_text)
        # else:
        #     st.write("### Extracted Text")
        #     st.write(website_text)

            with st.spinner("Evaluating text quality using Bedrock..."):
                evaluation_text = evaluate_text_quality_bedrock(website_text)

            st.write("### Evaluation Result")

            # Display the raw evaluation result from Bedrock
            st.write(evaluation_text)

    else:
        st.error("Please enter a valid URL.")

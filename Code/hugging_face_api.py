import requests
import re

# Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": "Bearer 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"}  # Replace with your actual API key

# Function to extract answers from the file
def extract_answers(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract answers using regex
    answers = re.findall(r"Answer \d+:\s*(.*)", content)
    return answers

# Function to get correctness scores (0 to 1) for all answers in **one batch API call**
def get_correctness_scores(answers):
    payload = {
        "inputs": answers,  # Sending all answers at once
        "parameters": {"candidate_labels": ["Correct", "Incorrect"]},
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return [0] * len(answers)  # Default to 0 if API fails

    results = response.json()
    
    # Extract the "Correct" confidence score for each answer (0 to 1)
    scores = [res["scores"][0] for res in results]  # "Correct" is always at index 0
    
    return scores

# Function to compute the final average correctness score
def compute_score(file_path):
    answers = extract_answers(file_path)
    total_questions = len(answers)

    correctness_scores = get_correctness_scores(answers)  # Get correctness scores (0 to 1)

    average_score = sum(correctness_scores) / total_questions if total_questions > 0 else 0.0

    return average_score, correctness_scores  # Return both the final score and per-answer scores

import pandas as pd
import json

# Load the dataset
file_path = 'enggcolleges.csv'
df = pd.read_csv(file_path)

# Initialize list for questions and contexts
qa_data = {"data": []}

# Loop through the dataset and create questions and context
for index, row in df.iterrows():
    # College-specific questions and answers
    college_name = row['College Name']
    fees = row['Average Fees']
    college_type = row['College Type']
    genders = row['Genders Accepted']

    # Question-Answer pairs
    questions_and_contexts = [
        {
            "question": f"What is the average fee for {college_name}?",
            "context": f"The average fee for {college_name} is {fees}."
        },
        {
            "question": f"Is {college_name} a public or private college?",
            "context": f"{college_name} is a {college_type} institution."
        },
        {
            "question": f"Does {college_name} accept both genders?",
            "context": f"{college_name} accepts {genders}."
        }
    ]

    qa_data["data"].extend(questions_and_contexts)

# Save the question-answer data to a JSON file
output_file = 'university_qa_data.json'
with open(output_file, 'w') as f:
    json.dump(qa_data, f, indent=4)

print(f"Question-answer dataset saved to {output_file}")

# Load the existing dataset
with open('university_qa_data.json', 'r') as f:
    data = json.load(f)

# Function to extract answer from context and add the 'answers' field
def add_answers(data):
    updated_data = {"data": []}
    
    for item in data["data"]:
        context = item["context"]
        question = item["question"]
        
        # Define rules for extracting answers based on question type
        if "fee" in question.lower():
            answer_text = context.split()[-1]  # Extract the last word (the fee)
        elif "public or private" in question.lower():
            answer_text = context.split()[-2]  # Extract the second last word
        elif "accept both genders" in question.lower():
            answer_text = "Co-Ed"  # Use predefined answer for gender question
        else:
            answer_text = "Unknown"  # Handle unknown cases
        
        # Find the starting position of the answer in the context
        start_position = context.find(answer_text)

        # Add the answers field
        updated_item = {
            "question": item["question"],
            "context": item["context"],
            "answers": {
                "text": [answer_text],
                "answer_start": [start_position]
            }
        }

        updated_data["data"].append(updated_item)

    return updated_data

# Add answers to the dataset
updated_data = add_answers(data)

# Save the updated dataset to a new JSON file
with open('university_qa_data_with_answers.json', 'w') as f:
    json.dump(updated_data, f, indent=4)

print("Dataset updated with answers.")

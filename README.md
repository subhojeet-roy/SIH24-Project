# SIH24-Project

University Admission Chatbot

This project is a Streamlit-based chatbot designed to assist students with university admission-related queries such as fee structure, eligibility criteria, scholarships, placements, and more. The chatbot uses a fine-tuned BERT model hosted on Hugging Face for Natural Language Processing (NLP) and Question Answering (QA) tasks.
Features

    Answer University Queries: Ask questions about university admissions, eligibility, fees, placements, scholarships, hostel facilities, and more.
    Fine-tuned BERT Model: The chatbot uses a fine-tuned BERT model trained on university data to provide accurate and relevant responses.
    Streamlit Web App: The chatbot is hosted on a simple and interactive web interface powered by Streamlit.
    GPU Support: The model is optimized to run on GPUs for faster inference (if available).

Demo

Live Demo Link
Replace with your deployed app link once available
How It Works

    User Input: The user inputs a question related to university admissions.
    Model Processing: The fine-tuned BERT model processes the question and generates an answer.
    Answer Output: The chatbot displays the answer in response to the user's query.

Installation and Setup
1. Clone the repository

bash

git clone https://github.com/subhojeet-roy/SIH24-Project.git
cd SIH24-Project

2. Install Dependencies

You need Python 3.7 or later installed. Install the required dependencies using pip:

bash

pip install -r requirements.txt

3. Run the Chatbot

To run the chatbot locally, use Streamlit:

bash

streamlit run app.py

This will launch the web application in your default browser.
Fine-Tuned Model

The BERT model used in this project was fine-tuned on a custom dataset containing university admission information. It is hosted on Hugging Face, and you can load it directly in your application.

    Model Repository: subhojeet-roy/SIH-Chatbot on Hugging Face

Dataset

The chatbot was trained on a dataset containing the following information about universities:

    Admission Process
    Eligibility Criteria
    Fee Structure
    Scholarships
    Hostel Facilities
    Placement Opportunities
    College-Specific Information

The dataset is structured in JSON format and was fine-tuned on the BERT model for Question Answering tasks.
Project Structure

bash

|-- app.py               # Main file for running the Streamlit chatbot
|-- fine_tune.py         # Script for fine-tuning the BERT model (if needed)
|-- university_qa_data.json  # Dataset containing the university information
|-- README.md            # Project documentation
|-- requirements.txt     # Python dependencies
|-- fine-tuned-bert/     # Directory containing the fine-tuned BERT model

Technologies Used

    Programming Language: Python
    Framework: Streamlit for the web interface
    Machine Learning: Hugging Face's transformers library, BERT model for Question Answering
    NLP: Fine-tuned BERT on custom university dataset
    Cloud Model Hosting: Hugging Face

Future Improvements

    Extend the dataset to cover more universities and programs.
    Enhance the chatbot's ability to handle complex queries.
    Add additional NLP techniques to improve accuracy and relevance.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features you'd like to add.
License

This project is licensed under the MIT License.
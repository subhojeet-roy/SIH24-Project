import streamlit as st
from transformers import pipeline

# Load the pre-trained question-answering model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Expanded context related to university admissions
context = """
The admission process for engineering and polytechnic colleges in Rajasthan involves an entrance exam and an online application form. 
The application deadline is typically in June. Scholarships are available for students from economically weaker sections and for meritorious students.

Fee Structure:
- The tuition fee for government engineering colleges is INR 50,000 per year.
- For private engineering colleges, the fee ranges between INR 80,000 to 1.5 Lakhs per year.

Scholarship Information:
- Scholarships are available based on merit and for students from economically weaker sections.
- The amount of scholarships ranges from INR 10,000 to full tuition fee waivers depending on the student's eligibility.

Previous Year's Allotments:
- In the previous year, students with a minimum score of 85% in their entrance exams were allotted seats in top government colleges.
"""

# Streamlit interface
st.title("University Admission Chatbot")
user_input = st.text_input("Ask your admission-related question:")

if user_input:
    # Get the answer from the model
    result = qa_pipeline({'question': user_input, 'context': context})
    st.write("Answer:", result['answer'])

from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer from Hugging Face
tokenizer = BertTokenizerFast.from_pretrained('subhojeet-roy/SIH-Chatbot')
model = BertForQuestionAnswering.from_pretrained('subhojeet-roy/SIH-Chatbot')

# Ensure the model is on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to answer questions
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move input tensors to GPU

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely start and end of the answer within the context
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert token indices to the actual answer string
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    return answer

# Example chatbot function that uses the fine-tuned model to respond
def chatbot_response(user_input):
    # Example context, you can modify this based on how you manage data in your chatbot
    context = "The average fee for National Institute of Technology Rourkela is 350600.0."
    
    # Get the model's answer
    answer = answer_question(user_input, context)
    
    # Return the answer
    return answer

# Example usage of the chatbot
if __name__ == '__main__':
    # Simulate a user question
    user_input = input("Ask a question: ")
    response = chatbot_response(user_input)
    print(f"Chatbot response: {response}")

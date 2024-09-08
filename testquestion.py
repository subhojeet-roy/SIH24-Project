from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer
model_dir = '/home/subhojeet-roy/fine-tuned-bert'
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForQuestionAnswering.from_pretrained(model_dir)

# Sample question and context
question = "What is the average fee for National Institute of Technology Rourkela?"
context = "The average fee for National Institute of Technology Rourkela is 350600.0."

# Tokenize the inputs
inputs = tokenizer(question, context, return_tensors='pt')

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

# Convert token IDs to the answer text
answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
answer = tokenizer.convert_tokens_to_string(answer_tokens)

# Clean up spaces and handle cases like "350600. 0." to "350600.0"
answer = answer.replace(" .", ".").replace(" ,", ",").replace(" 0.", "0.")  # Additional cleaning for number formatting

print(f"Answer: {answer}")

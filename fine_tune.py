from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Ensure the model is using the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the GPU

# Load the updated dataset
def load_custom_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    questions = [item['question'] for item in data['data']]
    contexts = [item['context'] for item in data['data']]
    answers = [item['answers'] for item in data['data']]  # Use 'answers' as a dictionary containing 'text' and 'answer_start'

    return {"question": questions, "context": contexts, "answers": answers}

# Load the updated dataset with answers
dataset = load_custom_dataset('university_qa_data_with_answers.json')

# Preprocess function to tokenize and handle GPU placement
def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True  # Needed for start/end position
    )
    
    # Move tokenized examples to GPU
    tokenized_examples = {key: torch.tensor(val).to(device) for key, val in tokenized_examples.items()}
    
    # Add answer positions
    return add_answer_positions(examples, tokenized_examples)

# Add the function for adding answer positions
def add_answer_positions(examples, tokenized_examples):
    start_positions = []
    end_positions = []

    for i, offset_mapping in enumerate(tokenized_examples["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]  # Start position of the answer
        end_char = start_char + len(answer["text"][0])  # End position of the answer

        # Get token start and end positions
        token_start, token_end = None, None
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                token_start = idx
            if start < end_char <= end:
                token_end = idx
                break

        if token_start is None or token_end is None:
            print(f"Error: Could not find start or end token for example {i}")
            token_start, token_end = 0, 0  # Default value to avoid crashes

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized_examples["start_positions"] = torch.tensor(start_positions).to(device)
    tokenized_examples["end_positions"] = torch.tensor(end_positions).to(device)

    return tokenized_examples

# Convert dataset into Hugging Face Dataset format
qa_dataset = Dataset.from_dict(dataset)

# Apply preprocessing to the dataset
tokenized_dataset = qa_dataset.map(preprocess_function, batched=True)

# Training arguments with GPU settings
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='no',
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # You can try 32 if memory allows
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1000,
    fp16=True,  # Enable mixed precision
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')

print("Model fine-tuned and saved!")


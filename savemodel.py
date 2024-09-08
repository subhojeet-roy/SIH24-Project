from transformers import BertTokenizerFast, BertForQuestionAnswering

# Path to the latest checkpoint
checkpoint_dir = './results/checkpoint-1022'  # Adjust if necessary

# Load model from the checkpoint
print(f"Loading model from checkpoint: {checkpoint_dir}")
model = BertForQuestionAnswering.from_pretrained(checkpoint_dir)  # Loading model from checkpoint

# Load base tokenizer since the checkpoint doesn't contain tokenizer files
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Using base tokenizer

# Save the model and tokenizer to the final directory
save_dir = '/home/subhojeet-roy/fine-tuned-bert'
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved successfully to {save_dir}")

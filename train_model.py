# -*- coding: utf-8 -*-
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

# Load the dataset
file_path = 'own_dataset.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the dataset
texts = [item['text'] for item in data]
summaries = [item['summary'] for item in data]

# Convert to Dataset format
dataset = Dataset.from_dict({'text': texts, 'summary': summaries})
dataset = dataset.train_test_split(test_size=0.1)

# Define the model and tokenizer
model_name = 'facebook/mbart-large-cc25'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['summary'], max_length=150, truncation=True, padding="max_length")
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./telugu_summary_model')
tokenizer.save_pretrained('./telugu_summary_model')

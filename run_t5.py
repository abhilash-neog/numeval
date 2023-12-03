# reference
# https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

import os
import json
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

nltk.download('punkt')

train_path = '/fastscratch/mridul/numeval/Train_Headline_Generation.json'
dev_path = '/fastscratch/mridul/numeval/Dev_Headline_Generation.json'

# with open(train_path, 'r') as f:
#     training_data = json.load(f)

# with open(dev_path, 'r') as f:
#     dev_data = json.load(f)

# train_dataset = load_dataset("json", data_files=train_path, split="train")
# dev_dataset = load_dataset("json", data_files=dev_path)


data_files = {'train': train_path, 'validation': dev_path}
ds = load_dataset("json", data_files=data_files)

# DatasetDict({
#     train: Dataset({
#         features: ['headline', 'news'],
#         num_rows: 21157
#     })
#     validation: Dataset({
#         features: ['headline', 'news'],
#         num_rows: 2365
#     })
# })

model_checkpoint = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

prefix = "summarize: "
max_input_length = 512
max_target_length = 32
device = 'cuda'

def preprocess_data(examples):
    inputs = [prefix + text for text in examples["news"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["headline"], max_length=max_target_length, 
                            truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = ds.map(preprocess_data, batched=True)

batch_size = 16
# model_name = "t5-small"
model_dir = f"/fastscratch/mridul/numeval/models/{model_checkpoint}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=800,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=800,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    # fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1"
    # report_to="tensorboard"
)


data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

print('Done')
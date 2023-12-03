import torch
import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

train_path = '/fastscratch/mridul/numeval/Train_Headline_Generation.json'
dev_path = '/fastscratch/mridul/numeval/Dev_Headline_Generation.json'

prefix = "summarize: "
max_input_length = 512
max_target_length = 32
device = 'cuda'

data_files = {'train': train_path, 'validation': dev_path}
ds = load_dataset("json", data_files=data_files)

# model_path = "/fastscratch/mridul/numeval/models/t5-small/checkpoint-1200"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# get test split
test_tokenized_dataset = ds["validation"]





# pad texts to the same length
def preprocess_test(examples):
  inputs = [prefix + text for text in examples["news"]]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,
                           padding="max_length")
  return model_inputs

test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True)


# prepare dataloader
test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

# generate text for each batch
all_predictions = []
for i,batch in enumerate(tqdm(dataloader)):
    # breakpoint()
    batch = {k: v.to(device) for k, v in batch.items()}
    predictions = model.generate(**batch)
    all_predictions.append(predictions.cpu())


# flatten predictions
all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

# tokenize and pad titles
all_titles = tokenizer(test_tokenized_dataset["headline"], max_length=max_target_length,
                       truncation=True, padding="max_length")["input_ids"]


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

# compute metrics
predictions_labels = [all_predictions_flattened, all_titles]
answers = compute_metrics(predictions_labels)

print(answers)
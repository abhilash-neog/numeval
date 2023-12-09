# reference
# https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

import os
import json
import nltk
import numpy as np
from datetime import datetime
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer, AutoConfig
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

nltk.download('punkt')
os.environ["WANDB_PROJECT"] = 'NLP'


train_path = '/fastscratch/mridul/numeval/Train_Headline_Generation.json'
dev_path = '/fastscratch/mridul/numeval/Dev_Headline_Generation.json'

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

prefix = "summarize: "
max_input_length = 512
max_target_length = 32
device = 'cuda'

model_checkpoint = "t5-3b"
model_save_name = 't5_3b_5e-5_epoch15_dropout0.3_warm50k'
run_name = f"{model_save_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
batch_size = 4
model_dir = f"/fastscratch/mridul/numeval/models/{run_name}"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def model_init():

    config = AutoConfig.from_pretrained(model_checkpoint)
    config.dropout = 0.3 
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config).to(device)



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

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=2400,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=2400,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    warmup_steps=50000,
    save_total_limit=3,
    num_train_epochs=25,
    predict_with_generate=True,
    # fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="wandb",
    run_name=run_name
)


data_collator = DataCollatorForSeq2Seq(tokenizer)

def compute_metrics(eval_pred):
    metric = load_metric("rouge")
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
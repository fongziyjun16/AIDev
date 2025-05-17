from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np

dataset = load_dataset("./datasets/yelp_review_full")
# print(len(dataset["train"]))
# print(len(dataset["test"]))

tokenizer = AutoTokenizer.from_pretrained("models/base/google-bert/bert-base-cased")
# print(tokenizers)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset["train"].shuffle(seed=32)
test_dataset = tokenized_dataset["test"].shuffle(seed=32)

model = AutoModelForSequenceClassification.from_pretrained(
    "models/base/google-bert/bert-base-cased",
    num_labels=5
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

fine_tuned_model_dir = "models/fine_tuned/bert-base-cased-ft-yelp"

training_args = TrainingArguments(
    output_dir=fine_tuned_model_dir,
    per_device_train_batch_size=16,
    num_train_epochs=4,
    logging_steps=500
)
training_args.eval_strategy = "epoch"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
print(trainer.evaluate())
trainer.save_model()

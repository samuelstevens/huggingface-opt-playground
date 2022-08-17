import datasets
from transformers import AutoTokenizer

checkpoint = "facebook/opt-30b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)


def tokenize_boolq(example):
    text = f"{example['passage']}\nquestion: {example['question']}\nanswer: "
    return tokenizer(text)


dataset = datasets.load_dataset("super_glue", "boolq", split="validation")

dataset = dataset.map(tokenize_boolq)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
acc = datasets.load_metric("accuracy")

breakpoint()

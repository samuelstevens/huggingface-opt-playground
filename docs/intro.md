# Intro to Using Huggingface with OPT

This is a basic example of using Huggingface's OPT for some cool tasks.

For example, extracting information from Wikipedia articles:

```sh
python -m src.extract --model facebook/opt-30b
```

## List of Models

* facebook/opt-30b
* facebook/opt-13b
* facebook/opt-6.7b
* facebook/opt-2.7b
* facebook/opt-1.3b
* facebook/opt-350m
* facebook/opt-125m
* gpt2-xl
* gpt2-large
* gpt2-medium
* gpt2

## Modeling

To fit the big models on multiple GPUs, we use huggingface accelerate in `src/modeling.py`
Some of the checkpoints are in different formats to each other when I originally worked on this script.

## Scripts

The scripts just run a particular python file for all models so we can graph the results with respect to model size.

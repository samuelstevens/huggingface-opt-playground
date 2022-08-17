"""
For each example in boolq, measure a model's perplexity on the following templates, then chooses the template with lower perplexity.

context: <passage>
question: <question>?
answer: yes

context: <passage>
question: <question>?
answer: no
"""
import argparse

import datasets
import torch
from tqdm.auto import tqdm

from . import helpers, modeling


def templated(example, answer) -> str:
    assert answer in ("yes", "no")

    passage = example["passage"]
    question = example["question"].rstrip("?")

    return f"context: {passage}\nquestion: {question}?\nanswer: {answer}"


def measure_perplexity(model, encodings, device) -> float:
    max_length = model.config.max_position_embeddings

    if encodings.input_ids.size(1) > max_length:
        return -1

    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    # Everything but last token is context.
    target_ids[:, :-1] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        return torch.exp(outputs[0]).item()


def main():
    parser = argparse.ArgumentParser()
    parser = helpers.add_model_arg(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dataset = datasets.load_dataset("super_glue", "boolq", split="validation")
    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(args.model)

    correct = 0
    total = len(dataset)

    for example in tqdm(dataset):
        yes_ppl = measure_perplexity(
            model, helpers.tokenize(templated(example, "yes"), tokenizer), device
        )

        no_ppl = measure_perplexity(
            model, helpers.tokenize(templated(example, "no"), tokenizer), device
        )

        if no_ppl < 0 or yes_ppl < 0:
            continue

        # 1 is true/yes, 0 is false/no
        pred = 1 if yes_ppl < no_ppl else 0

        if pred == example["label"]:
            correct += 1

    print(correct, total, correct / total)


if __name__ == "__main__":
    main()

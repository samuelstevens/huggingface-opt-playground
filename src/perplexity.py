"""
Evaluates a language model's perplexity using:
    * Model
    * Stride length
    * Dataset
"""

import argparse

import datasets
import torch
from tqdm.auto import tqdm

from . import helpers, modeling


def _load_wikitext():
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(dataset["text"])


def load_data(dataset_name):
    if dataset_name == "wikitext":
        return _load_wikitext()
    else:
        raise ValueError(dataset_name)


def measure_perplexity(model, stride, encodings, device):
    max_length = model.config.max_position_embeddings

    # negative log likelihoods
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood.item())

    return torch.exp(torch.tensor(nlls).sum() / end_loc).item()


def main():
    parser = argparse.ArgumentParser()
    parser = helpers.add_model_arg(parser)
    parser.add_argument(
        "--context", type=int, help="how long a context to use", required=True
    )
    parser.add_argument(
        "--dataset",
        choices=["wikitext"],
        help="which dataset to measure perplexity on",
        required=True,
    )

    args = parser.parse_args()

    encodings = helpers.tokenize(
        load_data(args.dataset), modeling.load_tokenizer(args.model)
    )

    print(
        measure_perplexity(
            modeling.load_model(args.model),
            args.context,
            encodings,
            torch.device("cuda:0"),
        )
    )


if __name__ == "__main__":
    main()

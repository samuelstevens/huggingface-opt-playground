"""
Tries to extract structured data from Wikipedia descriptions of birds.
"""

import argparse
import dataclasses
import json
from typing import List, Tuple

import torch

from . import helpers, modeling


def make_context(examples):
    """

    """
    example_contexts = "\n\n".join(example.as_context() for example in examples)
    return example_contexts


@dataclasses.dataclass
class Example:
    """
    An example has some physical description (self.physical) and some manually extracted traits, which are pairs of strings (self.traits).
    """
    physical: str
    traits: List[Tuple[str, str]]

    def as_context(self):
        """
        Returns the example as a context string for use with language model.
        """
        trait_contexts = [
            f"{feature}: {description}" for feature, description in self.traits
        ]

        trait_context = "\n".join(trait_contexts)

        return f"DESCRIPTION:\n{self.physical}\n\nTRAITS:\n{trait_context}"


def load_example(filepath: str) -> Example:
    """
    Reads some data from a .json file and makes an Example.
    """
    with open(filepath) as fd:
        raw = json.load(fd)

    physical = raw["physical"]
    traits = []
    if "traits" in raw:
        for key, value in raw["traits"].items():
            traits.append((key, value))

    return Example(physical, traits)


def main():
    parser = argparse.ArgumentParser()
    parser = helpers.add_model_arg(parser)
    args = parser.parse_args()

    examples = [
        load_example("data/wikipedia/blue-jay.json"),
        load_example("data/wikipedia/northern-cardinal.json"),
        load_example("data/wikipedia/american-robin.json"),
    ]

    device = torch.device("cuda:0")
    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(args.model)

    prompt = make_context(examples)

    max_model_length = model.config.max_position_embeddings
    max_output_length = 100

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding.input_ids[:, -(max_model_length - max_output_length) :]
    attn_mask = encoding.attention_mask[:, -(max_model_length - max_output_length) :]

    greedy_output = model.generate(
        input_ids.to(device),
        attention_mask=attn_mask.to(device),
        max_length=max_model_length,
    )
    input("Press <ENTER> to see output: ")
    print("Output:\n" + 100 * "-")
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    for trait in ("beak", "bill", "legs", "wings"):
        encoding = tokenizer(f"{prompt}{trait}: ", return_tensors="pt")
        max_output_length = 20
        input_ids = encoding.input_ids[:, -(max_model_length - max_output_length) :]
        attn_mask = encoding.attention_mask[
            :, -(max_model_length - max_output_length) :
        ]

        greedy_output = model.generate(
            input_ids.to(device),
            attention_mask=attn_mask.to(device),
            max_length=max_model_length,
        )
        input(f"Press <ENTER> to see output for {trait}: ")
        print("Output:\n" + 100 * "-")
        print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

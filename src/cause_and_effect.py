import argparse
import dataclasses
import json
from typing import Tuple

import torch
from tqdm.auto import tqdm

from . import helpers, modeling


def load_data():
    with open("data/cause_and_effect_two_sentences.json") as fd:
        data = json.load(fd)

    for example in data["examples"]:
        cause, effect = None, None
        choices = tuple(example["target_scores"].keys())
        for choice in choices:
            if example["target_scores"][choice] > 0:
                cause = choice
            else:
                effect = choice

        assert cause and effect
        assert cause != effect

        yield Example(choices, cause, effect)


@dataclasses.dataclass
class Example:
    choices: Tuple[str, str]
    cause: str
    effect: str

    @property
    def template(self):
        return f"For each example, two events are given. Which event caused the other?\nchoice: {self.choices[0]}\nchoice: {self.choices[1]}\nanswer: "


def measure_perplexity(template, answer, model, tokenizer):
    device = torch.device("cuda:0")

    template_ids = helpers.tokenize(template, tokenizer).input_ids
    answer_ids = helpers.tokenize(answer, tokenizer).input_ids
    answer_len = answer_ids.size(1)

    input_ids = torch.cat((template_ids, answer_ids), 1).to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-answer_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        return torch.exp(outputs[0]).item()


def main():
    parser = argparse.ArgumentParser()
    parser = helpers.add_model_arg(parser)
    args = parser.parse_args()

    tokenizer = modeling.load_tokenizer(args.model)
    model = modeling.load_model(args.model)

    data = list(load_data())

    total = len(data)
    correct = 0

    for example in tqdm(data):
        cause_ppl = measure_perplexity(
            example.template, example.cause, model, tokenizer
        )
        effect_ppl = measure_perplexity(
            example.template, example.effect, model, tokenizer
        )

        if cause_ppl < effect_ppl:
            correct += 1

    print(correct, total, correct / total)


if __name__ == "__main__":
    main()

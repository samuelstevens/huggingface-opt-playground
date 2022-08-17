"""
Plots the perplexity in results.json.
"""
import argparse
import json

# Parameter counts in millions
param_counts = {
    # https://github.com/openai/gpt-2/blob/master/DEVELOPERS.md
    # For GPT2 sizes
    "gpt2": 124,
    "gpt2-medium": 355,
    "gpt2-large": 774,
    "gpt2-xl": 1_500,
    "facebook/opt-125m": 125,
    "facebook/opt-350m": 350,
    "facebook/opt-1.3b": 1_300,
    "facebook/opt-2.7b": 2_700,
    "facebook/opt-6.7b": 6_700,
    "facebook/opt-13b": 13_000,
    "facebook/opt-30b": 30_000,
}

MILLION = 1_000_000


def plot_perplexity():
    with open("results/perplexity.json") as fd:
        perplexities = json.load(fd)

    perplexities = {
        key.removesuffix("-512-wikitext"): value for key, value in perplexities.items()
    }

    # GPT2s
    gpt_xs, gpt_ys = [], []
    for key in perplexities:
        if "gpt" not in key:
            continue

        gpt_xs.append(param_counts[key] * MILLION)
        gpt_ys.append(perplexities[key])

    # OPTs
    opt_xs, opt_ys = [], []
    for key in perplexities:
        if "opt" not in key:
            continue

        opt_xs.append(param_counts[key] * MILLION)
        opt_ys.append(perplexities[key])

    import matplotlib.pyplot as plt

    plt.plot(gpt_xs, gpt_ys, label="GPT2")
    plt.plot(opt_xs, opt_ys, label="OPT")
    plt.xscale("log")
    plt.xlabel("Parameters")
    plt.ylabel("Perplexity")
    plt.title("GPT2 vs OPT Perplexity on Wikitext")
    plt.legend()
    plt.savefig("plots/perplexities.pdf")


def plot_jsonl(
    in_filename,
    out_filename,
    title,
    x_factor=MILLION,
    y_factor=100,
    ylabel="Accuracy",
):
    with open(f"results/{in_filename}") as fd:
        lines = [json.loads(line.strip()) for line in fd]

    # GPT2s
    gpt_xs, gpt_ys = [], []
    for line in sorted(lines, key=lambda line: param_counts[line["checkpoint"]]):
        if "gpt" not in line["checkpoint"]:
            continue

        gpt_xs.append(param_counts[line["checkpoint"]] * x_factor)
        gpt_ys.append(line["result"] * y_factor)

    # OPTs
    opt_xs, opt_ys = [], []
    for line in sorted(lines, key=lambda line: param_counts[line["checkpoint"]]):
        if "opt" not in line["checkpoint"]:
            continue

        opt_xs.append(param_counts[line["checkpoint"]] * x_factor)
        opt_ys.append(line["result"] * y_factor)

    import matplotlib.pyplot as plt

    plt.plot(gpt_xs, gpt_ys, label="GPT2", marker="o")
    plt.plot(opt_xs, opt_ys, label="OPT", marker="o")
    plt.xscale("log")
    plt.xlabel("Parameters")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{out_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", choices=["perplexity", "boolq", "cause-and-effect"])

    args = parser.parse_args()

    if args.graph == "perplexity":
        plot_perplexity()
    elif args.graph == "boolq":
        plot_jsonl(
            "boolq.jsonl",
            "boolq.pdf",
            "BoolQ Zero-Shot Acc.",
        )
    elif args.graph == "cause-and-effect":
        plot_jsonl(
            "cause_and_effect.jsonl",
            "cause_and_effect.pdf",
            "Cause & Effect (BIG-Bench) Zero-Shot Acc.",
        )
    else:
        raise ValueError(args.graph)


if __name__ == "__main__":
    main()

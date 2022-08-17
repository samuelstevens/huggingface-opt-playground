import contextlib
import sys


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@contextlib.contextmanager
def logged(msg):
    log(f"Starting: [{msg}]")
    try:
        yield
    finally:
        log(f"Finished: [{msg}]")


def tokenize(text, tokenizer):
    return tokenizer(text, return_tensors="pt")


def add_model_arg(parser):
    parser.add_argument(
        "--model",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
        ],
        help="which model to evaluate",
        required=True,
    )
    return parser

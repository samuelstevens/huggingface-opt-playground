echo $CUDA_VISIBLE_DEVICES

run () {
  jo checkpoint=$1 result=$(python -m src.boolq --model $1 | awk '{print $3}') >> results/boolq.jsonl
}

# run facebook/opt-30b
# run facebook/opt-13b
# run facebook/opt-6.7b
# run facebook/opt-2.7b
# run facebook/opt-1.3b
# run facebook/opt-350m
# run facebook/opt-125m
run gpt2-xl
# run gpt2-large
# run gpt2-medium
# run gpt2

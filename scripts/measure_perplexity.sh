echo $CUDA_VISIBLE_DEVICES

# python -m src.perplexity --model facebook/opt-30b --context 512 --dataset wikitext > opt-30b-512-wikitext
# python -m src.perplexity --model facebook/opt-13b --context 512 --dataset wikitext > opt-13b-512-wikitext
# python -m src.perplexity --model facebook/opt-6.7b --context 512 --dataset wikitext > opt-6.7b-512-wikitext
python -m src.perplexity --model facebook/opt-2.7b --context 512 --dataset wikitext > opt-2.7b-512-wikitext
python -m src.perplexity --model facebook/opt-1.3b --context 512 --dataset wikitext > opt-1.3b-512-wikitext
# python -m src.perplexity --model facebook/opt-350m --context 512 --dataset wikitext > opt-350m-512-wikitext
python -m src.perplexity --model facebook/opt-125m --context 512 --dataset wikitext > opt-125m-512-wikitext
# python -m src.perplexity --model gpt2-large --context 512 --dataset wikitext > gpt2-large-512-wikitext
# python -m src.perplexity --model gpt2-medium --context 512 --dataset wikitext > gpt2-medium-512-wikitext
# python -m src.perplexity --model gpt2 --context 512 --dataset wikitext > gpt2-512-wikitext

set -e

topks=("0.2" "0.3" "0.4" "0.5")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Calculating optimal and model perplexity for gpt2"
        python eval.py --model gpt2 \
                        --version $version \
                        --top_k $topk \
                        --size 100 \
                        --device cpu
    done
done

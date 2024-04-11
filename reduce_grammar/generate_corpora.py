from classes import (PCFG, PCFGConfig,
                     TokenizerConfig, Tokenizer)
from nltk import PCFG as nltk_PCFG
import logging
import argparse


def main(args):
    grammar_file = f'{args.data_dir}/{args.version}/subset_pcfg_{args.top_k}.txt'
    encoder = "transformer"
    tokenizer_config = TokenizerConfig(
            add_cls=(encoder == "transformer"),
            masked_lm=(encoder == "transformer"),
            unk_threshold=5,
        )

    config = PCFGConfig(
        is_binary=False,
        min_length=args.min_length,
        max_length=args.max_length,
        max_depth=args.max_depth,
        corpus_size=args.corpus_size,
        grammar_file=grammar_file,
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(0.8,0.1,0.1),
        use_unk_pos_tags=True,
        verbose=args.verbose,
        store_trees=False,
        output_dir=args.output_dir,
        top_k=args.top_k,
        version=args.version,
        file=args.corpus_file
    )
    tokenizer = Tokenizer(tokenizer_config)
    
    lm_language = PCFG(config, tokenizer)
    lm_language.save_pcfg()
    lm_language.save(f'{args.output_dir}/corpus_{args.top_k}_{args.version}.pt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate corpora')
    parser.add_argument('--version', type=str, default='normal', choices=['normal', 'lexical', 'pos'],
                        help='Version of the corpus to generate.')
    parser.add_argument('--data_dir', type=str, default='grammars/nltk')
    parser.add_argument('--output_dir', type=str, default='corpora',
                        help='Output directory for the generated corpora.')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--corpus_size', type=int, default=1_000_000,
                        help='Size of the corpus to generate.')
    parser.add_argument('--min_length', type=int, default=6)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--max_depth', type=int, default=25)
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--top_k', type=float, default=0.2)

    args = parser.parse_args()

    main(args)

        



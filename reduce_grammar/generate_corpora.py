from classes import (PCFG, PCFGConfig,
                     TokenizerConfig, Tokenizer)
from nltk import PCFG as nltk_PCFG
import logging
import argparse




def main(args):
    for top_k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        grammar_file = f'grammars/nltk/{args.version}/subset_pcfg_{top_k}.txt'
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
            store_trees=True,
            output_dir=args.output_dir,
            top_k=top_k,
            version=args.version
        )
        tokenizer = Tokenizer(tokenizer_config)
        
        lm_language = PCFG(config, tokenizer)
        lm_language.save_pcfg()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate corpora')
    parser.add_argument('--version', type=str, default='normal', choices=['normal', 'lexical', 'pos'],
                        help='Version of the corpus to generate.')
    parser.add_argument('--output_dir', type=str, default='corpora',
                        help='Output directory for the generated corpora.')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--corpus_size', type=int, default=1_000_000,
                        help='Size of the corpus to generate.')
    parser.add_argument('--min_length', type=int, default=6)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--max_depth', type=int, default=25)

    args = parser.parse_args()

    main(args)

        



from classes import (PCFG, PCFGConfig,
                     TokenizerConfig, Tokenizer)
from nltk import PCFG as nltk_PCFG
import logging




def main():
    for top_k in [0.7, 0.8]:
        grammar_file = f'grammars/nltk/normal/subset_pcfg_{top_k}.txt'
        encoder = "transformer"
        tokenizer_config = TokenizerConfig(
                add_cls=(encoder == "transformer"),
                masked_lm=(encoder == "transformer"),
                unk_threshold=5,
            )

        config = PCFGConfig(
            is_binary=False,
            min_length=6,
            max_length=25,
            max_depth=25,
            corpus_size=100,
            grammar_file=grammar_file,
            start="S_0",
            masked_lm=(encoder == "transformer"),
            allow_duplicates=True,
            split_ratio=(0.8,0.1,0.1),
            use_unk_pos_tags=True,
            verbose=True,
            store_trees=True,
            output_dir='corpora/normal',
            top_k=top_k
        )
        tokenizer = Tokenizer(tokenizer_config)
        
        lm_language = PCFG(config, tokenizer)
        lm_language.save_pcfg()
        break

if __name__ == '__main__':
    main()

        



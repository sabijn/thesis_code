from classes import (PCFG, PCFGConfig,
                     TokenizerConfig, Tokenizer)
from nltk import PCFG as nltk_PCFG




if __name__ == '__main__':
    with open('grammars/nltk/nltk_pcfg.txt') as f:
        raw_grammar = f.read()
    grammar = nltk_PCFG.fromstring(raw_grammar)
    print(grammar._lhs_prob_index)

    # This part is usefull IF you have your subset pcfg file, you can create your corpus. 
    # Where to add the pruning
    # grammar_file = 'grammars/ntlk/nltk_pcfg.txt'
    # encoder = "lstm"
    # tokenizer_config = TokenizerConfig(
    #         add_cls=(encoder == "transformer"),
    #         masked_lm=(encoder == "transformer"),
    #         unk_threshold=5,
    #     )

    # config = PCFGConfig(
    #     is_binary=False,
    #     min_length=6,
    #     max_length=25,
    #     max_depth=25,
    #     corpus_size=9_500_000,
    #     grammar_file=grammar_file,
    #     start="S_0",
    #     masked_lm=(encoder == "transformer"),
    #     allow_duplicates=True,
    #     split_ratio=(1.,0.,0.),
    #     use_unk_pos_tags=True,
    #     verbose=True,
    #     store_trees=False,
    # )
    # tokenizer = Tokenizer(tokenizer_config)
    
    # lm_language = PCFG(config, tokenizer)



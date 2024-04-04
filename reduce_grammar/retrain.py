from classes import (PCFG, PCFGConfig,
                    TokenizerConfig, Tokenizer, 
                    ModelConfig, LanguageClassifier,
                    ExperimentConfig, Experiment)
from copy import deepcopy
from utils import plot_results

if __name__ == '__main__':
    test_scores = {}


    ratio = 1/10; print(ratio)

    grammar_file = f"petrov/500k_5SM/subsets/{ratio}_petrov_words_all_leaves.txt"
    encoder = 'transformer'

    tokenizer_config = TokenizerConfig(
        add_cls=(encoder == "transformer"),
        masked_lm=(encoder == "transformer"),
        unk_threshold=5,
    )
    tokenizer = Tokenizer(tokenizer_config)

    config = PCFGConfig(
        is_binary=False,
        min_length=6,
        max_length=20,
        max_depth=100,
        corpus_size=10_000,
        grammar_file=grammar_file,
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(.8,.1,.1),
        use_unk_pos_tags=True,
    )
    
    lm_language = PCFG(config, tokenizer)

    ## LM PRETRAINING
    model_config = ModelConfig(
        nhid = 25,
        num_layers = 2,
        vocab_size = len(tokenizer.idx2token),
        is_binary = False,
        encoder = encoder,
        num_heads = 3,
        one_hot_embedding = False,
        emb_dim = 25,
        learned_pos_embedding = True,
        pad_idx = tokenizer.pad_idx,
        mask_idx = tokenizer.mask_idx,
        non_linear_decoder = True,
    )
    model = LanguageClassifier(model_config)

    experiment_config = ExperimentConfig(
        lr=1e-2,  #tune.loguniform(1e-4, 1e-1),   # <- for lstm 1e-2 seems optimal often
        batch_size=48,  # tune.choice([32, 48, 64]),
        epochs=50,
        verbose=True,
        continue_after_optimum=0,
        eval_every=100,
        warmup_duration=0,
        early_stopping=1000,
        eval_dev_pos_performance=False,
        eval_test_pos_performance=True,
    )

    experiment = Experiment(
        model,
        experiment_config,
    )

    performance = experiment.train(lm_language)
    base_model = experiment.best_model
    model = deepcopy(base_model)  # detach reference so base_model can be used later
    plot_results(performance[0]['train'], performance[0]['dev'], performance[0]['test'], real_output=True)
    print(performance[0]['test'])

    ## BINARY CLASSIFICATION
    # model.is_binary = True
    # experiment_config.epochs = 100
    # experiment = Experiment(
    #     model,
    #     experiment_config,
    # )
    # performance = experiment.train(binary_language)
    # # plot_results(*performance[0])
    # plot_results(performance['train'], performance['dev'], performance['test'], real_output=True)

    # print(performance[0][2])
    # test_scores[ratio] = performance[0][2]
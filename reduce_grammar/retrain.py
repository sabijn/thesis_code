from classes import (PCFG, PCFGConfig,
                    TokenizerConfig, Tokenizer, 
                    ModelConfig, LanguageClassifier,
                    ExperimentConfig, Experiment)
from copy import deepcopy
from utils import plot_results
import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain')
    parser.add_argument('--data_dir', type=str, default='grammars/nltk')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--corpus_size', type=int, default=1_000_000,
                        help='Size of the corpus to generate.')
    parser.add_argument('--min_length', type=int, default=6)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--max_depth', type=int, default=25)
    parser.add_argument('--top_k', type=float, default=0.2)

    args = parser.parse_args()

    for top_k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        test_scores = {}

        grammar_file = f"{args.data_dir}/normal/subset_pcfg_{top_k}.txt"
        encoder = 'transformer'

        print('Loading tokenizer')
        tokenizer_config = TokenizerConfig(
            add_cls=(encoder == "transformer"),
            masked_lm=(encoder == "transformer"),
            unk_threshold=5,
        )
        tokenizer = Tokenizer(tokenizer_config)

        print('Initializing PCFG')
        config = PCFGConfig(
            is_binary=False,
            min_length=args.min_length,
            max_length=args.max_length,
            max_depth=100,
            corpus_size=args.corpus_size,
            grammar_file=grammar_file,
            start="S_0",
            masked_lm=(encoder == "transformer"),
            allow_duplicates=True,
            split_ratio=(.8,.1,.1),
            use_unk_pos_tags=True,
            file=args.corpus_file
        )

        lm_language = PCFG(config, tokenizer)

        ## LM PRETRAINING
        print('Initializing model')
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

        print("Start training")
        performance = experiment.train(lm_language)
        # dump performance in pkl
        with open(f"{args.output_dir}/performance.pkl", 'wb') as f:
            pickle.dump(performance, f)

        base_model = experiment.best_model
        model = deepcopy(base_model)  # detach reference so base_model can be used later
        # save model
        model.save(f"{args.output_dir}/model.pt")

        plot_results(args, top_k, performance[0]['train'], performance[0]['dev'], performance[0]['test'], real_output=True)
        break

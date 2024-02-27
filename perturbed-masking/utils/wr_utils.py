import pickle

def write_im_to_file(out, config):
    """
    Write the impact matrices per layer to a seperate file
    """
    for k, one_layer_out in enumerate(out):
        k_output = config.output_dir / f'{config.model}_{config.metric}_{str(k)}.pkl'
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()

def listtree2str(tree):
    """
    Convert tree represented as nested lists to bracketed string.
    """
    if isinstance(tree, str):
        return tree
    else:
        return '(' + ' '.join([listtree2str(subtree) for subtree in tree]) + ')'
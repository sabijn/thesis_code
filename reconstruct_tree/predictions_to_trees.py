from email.utils import parseaddr
import time
from tree2labels import SeqTree, RelativeLevelTreeEncoder
from nltk.tree import *
import argparse

def sequence_to_parenthesis(sentences,labels,join_char="~", split_char="@"):
#def sequence_to_parenthesis(sentences,labels,join_char="+"):
    parenthesized_trees = []  
    relative_encoder = RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char)
    
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse
    
    total_posprocessing_time = 0
    for noutput, output in enumerate(labels):       
        if output != "": #We reached the end-of-file
            init_parenthesized_time = time.time()
            sentence = []
            preds = []
            for ((word,postag), pred) in zip(sentences[noutput][1:-1],output[1:-1]):

                if len(pred.split(split_char))==3: #and "+" in pred.split("_")[2]:
                    sentence.append((word,pred.split(split_char)[2]+join_char+postag))             
                              
                else:
                    sentence.append((word,postag)) 
                        
#                 if len(pred.split("_"))==3: #and "+" in pred.split("_")[2]:
#                     sentence.append((word,pred.split("_")[2]+"+"+postag))                             
#                 else:
#                     sentence.append((word,postag)) 
#                 
                #TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
#                 if len(output)==3 and output[1] == "ROOT":
#                     pred = "NONE"     
        
                preds.append(pred)
   #         print preds
   #         print sentence
            tree = f_max_in_common(preds, sentence, relative_encoder)
                        
            #Removing empty label from root
            if tree.label() == SeqTree.EMPTY_LABEL:
                
                #If a node has more than two children
                #it means that the constituent should have been filled.
                if len(tree) > 1:
                    print ("WARNING: ROOT empty node with more than one child")
                else:
                    while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                        tree = tree[0]

            #Uncollapsing the root. Rare needed
            if join_char in tree.label():
                aux = SeqTree(tree.label().split(join_char)[0],[])
                aux.append(SeqTree(join_char.join(tree.label().split(join_char)[1:]), tree ))
                tree = aux
#             if "+" in tree.label():
#                 aux = SeqTree(tree.label().split("+")[0],[])
#                 aux.append(SeqTree("+".join(tree.label().split("+")[1:]), tree ))
#                 tree = aux
            tree = f_uncollapse(tree)
            

            total_posprocessing_time+= time.time()-init_parenthesized_time
            #To avoid problems when dumping the parenthesized tree to a file
            aux = tree.pformat(margin=100000000)
            
            if aux.startswith("( ("): #Ad-hoc workarounf for sentences of length 1 in German SPRML
                aux = aux[2:-1]
            parenthesized_trees.append(aux)

    return parenthesized_trees 

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-text')
    parser.add_argument('-lca')
    parser.add_argument('-levels')
    parser.add_argument('-out')
    parser.add_argument('-pos_text')
    parser.add_argument('-unary')
    parsedargs = parser.parse_args()
    

    # python3 predictions_to_trees.py -text ../neurox_syntax_experiments/2022_01_adjacent_lca/dev_orig_text.txt -lca ../neurox_syntax_experiments/2022_01_adjacent_lca/dev_rel_labels.txt -levels ../neurox_syntax_experiments/2022_01_adjacent_lca/roberta-base/regress_levels/16_preds_dev.txt -out pred_trees.txt -pos ../neurox_syntax_experiments/2022_01_adjacent_lca/dev_orig_pos.txt
    postextfile = parsedargs.pos_text
    lca_preds = parsedargs.lca
    level_preds = parsedargs.levels
    unaryfile = parsedargs.unary
    # posfile = parsedargs.pos

    with open(postextfile,'r') as f:
        text_and_pos = f.read().splitlines()
        wordsandpos = []
        w_p_sent = [('-BOS-','-BOS-')]
        for line in text_and_pos:
            if len(line) == 0:
                w_p_sent.append(('-EOS-','-EOS-'))
                wordsandpos.append(w_p_sent)
                w_p_sent = [('-BOS-','-BOS-')]
                continue
            [w,p] = line.split()
            w_p_sent.append((w,p))
    with open(lca_preds, 'r') as f:
        labels = f.read().splitlines()
        labels = [l.split() for l in labels]
    with open(level_preds, 'r') as f:
        levels = f.read().splitlines()
        levels = [l.split() for l in levels]
    with open(unaryfile, 'r') as f:
        unaries = f.read().splitlines()
        unaries = [l.split() for l in unaries]
    # with open(posfile, 'r') as f:
    #    poss = f.read().splitlines()
    #    poss = [l.split() for l in poss]
    assert len(wordsandpos) == len(labels) == len(levels) == len(unaries)
    assert all([len(l2) == len(l3) == len(l4) for l2,l3,l4 in zip(labels,levels,unaries)])
    assert all([len(l1) == (len(l2)+3) == (len(l3)+3) == len(l4)+3 for l1,l2,l3,l4 in zip(wordsandpos,labels,levels,unaries)])

    # pairs of tokens, pos
    # wordsandpos = [list(zip(ts,pos)) for ts,pos in zip(text,poss)]
    # wordsandpos = []
    # for txtline,posline in zip(text,poss):
    #     wordsandpos.append([('-BOS-','-BOS-')] + [[w,p] for w,p in zip(txtline,posline)] + [('-EOS-','-EOS-')])
    predseqs =  []
    for levs,lcas,uns in zip(levels,labels,unaries):
        predseq = ['-BOS-']
        for lev,lca,un in zip(levs,lcas,uns):
            if un!='XX':
                predseq.append(lev+';'+lca+';'+un)
            else:
                predseq.append(lev+';'+lca)
        predseq.append('NONE')
        predseq.append('-EOS-')
        predseqs.append(predseq)
        # predseqs.append( ['-BOS-'] + [lev+';'+lca for lev,lca in zip(levs,lcas)] + ['NONE','-EOS-'])
    
    print(wordsandpos[3])
    print(predseqs[3])
    # compute result
    res = sequence_to_parenthesis(wordsandpos, predseqs, split_char=';')
    
    # output
    with open(parsedargs.out, 'w') as f:
        f.writelines('\n'.join(res))
    print('done')

if __name__=='__main__':
    main()


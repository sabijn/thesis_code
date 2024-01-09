
from nltk.tree import Tree
import copy

"""
Class to manage the transformation of a constituent tree into a sequence of labels
and vice versa. It extends the Tree class from the NLTK framework to address constituent Parsing as a 
sequential labeling problem.
"""
class SeqTree(Tree):
    
    EMPTY_LABEL = "EMPTY-LABEL"
    
    def __init__(self,label,children):
         
        self.encoding = None
        super(SeqTree, self).__init__(label,children) 
    
    #TODO: At the moment only the RelativeLevelTreeEncoder is supported
    def set_encoding(self, encoding):
        self.encoding = encoding

    """
    Transforms a predicted sequence into a constituent tree
    @params sequence: A list of the predictions 
    @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
    @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
    concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
    """
    @classmethod
    def maxincommon_to_tree(cls, sequence, sentence, encoding):
        if encoding is None: raise ValueError("encoding parameter is None")
        return encoding.maxincommon_to_tree(sequence, sentence)

    """
    Gets the path from the root to each leaf node
    Returns: A list of lists with the sequence of non-terminals to reach each 
    terminal node
    """
    def path_to_leaves(self, current_path, paths):

        for i, child in enumerate(self):
            
            pathi = []
            if isinstance(child,Tree):
                common_path = copy.deepcopy(current_path)
                
                #common_path.append(child.label())
                common_path.append(child.label()+"*"+str(i))
                #common_path.append(child.label()+"-"+str(i))
                child.path_to_leaves(common_path, paths)
            else:
                for element in current_path:
                    pathi.append(element)
                pathi.append(child)
                paths.append(pathi)
    
        return paths
    
class RelativeLevelTreeEncoder(object):
    """
    Encoder/Decoder class to transform a constituent tree into a sequence of labels by representing
    how many levels in the tree there are in common between the word_i and word_(i+1) (in a relative scale) 
    and the label (constituent) at that lowest ancestor.
    """
    ROOT_LABEL = "ROOT"
    NONE_LABEL = "XX"

    SPLIT_LABEL_SURNAME_SYMBOL = "*"

    def __init__(self, join_char="~",split_char="@"):
        self.join_char = join_char
        self.split_char = split_char

    def uncollapse(self, tree):
        """
        Uncollapses the INTERMEDIATE unary chains and also removes empty nodes that might be created when
        transforming a predicted sequence into a tree.
        @precondition: Uncollapsing/Removing-empty from the root must be have done prior to to call 
        this function
        """
        uncollapsed = []
        for child in tree:

            if type(child) == type(u'') or type(child) == type(""):
                uncollapsed.append(child)
            else:
                #It also removes EMPTY nodes
                while child.label() == SeqTree.EMPTY_LABEL and len(child) != 0:
                    child = child[-1]
                
                label = child.label()
                #NEWJOINT
                if self.join_char in label:
                #if '+' in label: #and label[-1] != "+": #To support SPMRL datasets
                     
                    #NEWJOINT
                    label_split = label.split(self.join_char) 
                    #label_split = label.split('+')
                    swap = Tree(label_split[0],[])

                    last_swap_level = swap
                    for unary in label_split[1:]:
                        last_swap_level.append(Tree(unary,[]))
                        last_swap_level = last_swap_level[-1]
                    last_swap_level.extend(child)
                    uncollapsed.append(self.uncollapse(swap))
                #We are uncollapsing the child node
                else:     
                    uncollapsed.append(self.uncollapse(child))
        
        tree = Tree(tree.label(),uncollapsed)
        return tree
    
    
    """
    Gets a list of the PoS tags from the tree
    @return A list containing the PoS tags
    """
    def get_postag_trees(self,tree):
        
        postags = []
        
        for nchild, child in enumerate(tree):
            
            if len(child) == 1 and type(child[-1]) == type(""):
                postags.append(child)
            else:
                postags.extend(self.get_postag_trees(child))
        
        return postags

    def preprocess_tags(self,pred):
        """
        Transforms a prediction of the form LABEL_LEVEL_UNARY_CHAIN into a tuple
        of the form (level,label):
        level is an integer or None (if the label is NONE or NONE_leafunarychain).
        label is the constituent at that level
        @return (level, label)
        """
        pred = pred.split(self.split_char)
        label, level, unary = pred[0], pred[1], pred[2]

        if unary != self.NONE_LABEL:
            # unary leave should be incorporated into the label
            label = unary
            level = None # maybe necessary, check later
            return (level, label)
        
        if level == self.ROOT_LABEL:
            # shared level is the root itself
            return (level, label)

        return (int(level), label)
         
#     def preprocess_tags(self,pred):
        
#         try:         
#             #NEWJOINT
#            # label = pred.split("_")
#             label = pred.split(self.split_char)
#             level, label = label[0],label[1]  
#             try:
#                 return (int(level), label)
              # OWN UNDERSTANDING: this happens if the level is NONE or ROOT
              # and label is not None as this is a unary chain
#             except ValueError:
                
#                 #It is a NONE level with a leaf unary chain (e.g. NONE_ADJP)
#                 if level == self.NONE_LABEL:
#                     return (None,pred.rsplit(self.split_char,1)[1])
                
#                 return (level,label)
            
#         except IndexError:
#             # It is a NONE label without a leaf unary chain (e.g. NONE), pred is here also None
#             return (None, pred)
        
    def maxincommon_to_tree(self, sequence, sentence):
        """
        Transforms a predicted sequence into a constituent tree
        @params sequence: A list of the predictions 
        @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
        @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
        concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
        """
        tree = SeqTree(SeqTree.EMPTY_LABEL,[])
        current_level = tree
        previous_at = None
        first = True

        sequence = list(map(self.preprocess_tags, sequence))
        sequence = self._to_absolute_encoding(sequence)

        for j,(level,label) in enumerate(sequence):

            if level is None:
                prev_level, _ = sequence[j-1]
                previous_at = tree
                while prev_level is not None and prev_level > 1:
                    previous_at = previous_at[-1]
                    prev_level-=1
          
                #TODO: Trying optimitization
                #It is a NONE label
                if self.NONE_LABEL == label: #or self.ROOT_LABEL:
                #if "NONE" == label:
             #       if printing: print "ENTRA 1"
                    previous_at.append( Tree( sentence[j][1],[ sentence[j][0]]) )
                #It is a leaf unary chain
                #NEWJOINT
                else:
             #       if printing: 
             #           print "ENTRA 2"
             #          print "label", label
             #           print sentence[j]
                    
                    if label[0].isdigit() and self.ROOT_LABEL in label:
                  #      return Tree(self.join_char+sentence[j][1],[ sentence[j][0]])
                        previous_at.append(Tree(self.join_char+sentence[j][1],[ sentence[j][0]]))
            #            if printing: 
            #                print Tree(self.join_char+sentence[j][1],[ sentence[j][0]])
            #                print Tree(self.join_char+sentence[j][1],[ sentence[j][0]]).pretty_print()
                    else:
                        previous_at.append(Tree(label+self.join_char+sentence[j][1],[ sentence[j][0]]))   
#                else:
#                    previous_at.append(Tree(label+"+"+sentence[j][1],[ sentence[j][0]]))   

           #     if printing: exit()
                return tree
                continue
                   
            i=0
            for i in range(level-1):
      #      for i in xrange(level-1):
                if len(current_level) == 0 or i >= sequence[j-1][0]-1: 
                    child_tree = Tree(SeqTree.EMPTY_LABEL,[])                      
                    current_level.append(child_tree)   
                    current_level = child_tree

                else:
                    current_level = current_level[-1]
                    
            if current_level.label() == SeqTree.EMPTY_LABEL:    
                current_level.set_label(label)
                        
            if first:
                previous_at = current_level
                previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))
                first=False
            else:         
                #If we are at the same or deeper level than in the previous step
                if i >= sequence[j-1][0]-1: 
                    current_level.append(Tree( sentence[j][1],[sentence[j][0]]))
                else:
                    previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))        
                previous_at = current_level
                
            current_level = tree
            
        return tree

    def _tag(self,level,tag):
        #NEWJOINT
        return str(level)+self.split_char+tag.rsplit("*",1)[0]
      #  return str(level)+"_"+tag.rsplit("-",1)[0]
    
    def _to_absolute_encoding(self, relative_sequence):
        """
        Convert relative sequence (i.e. level, label per word) to absolute sequence.
        
        @params relative_sequence: list of tuples (level, label) per word
        Returns: 
        """
        absolute_sequence = [0] * len(relative_sequence)
        current_level = 0
        for j, (level,phrase) in enumerate(relative_sequence):
            if level is None:
                absolute_sequence[j] = (level, phrase)

            elif level == self.ROOT_LABEL:
                # phrase 
                aux_level = 1
                absolute_sequence[j] = (aux_level, phrase)
                current_level = aux_level
            
            else:
                current_level += level
                absolute_sequence[j] = (current_level, phrase)

        return absolute_sequence

    # def _to_absolute_encoding(self, relative_sequence):
        
    #     absolute_sequence = [0]*len(relative_sequence)
    #     current_level = 0
    #     for j,(level,phrase) in enumerate(relative_sequence):
        
    #         if level is None:
    #             absolute_sequence[j] = (level,phrase)
    #         elif type(level) == type("") and self.ROOT_LABEL in level:
                
    #        #     print level, level.replace(self.ROOT_LABEL,"")
    #             try:
    #                 aux_level = int(level.replace(self.ROOT_LABEL,""))
    #                 absolute_sequence[j] = (aux_level, phrase)
    #             except ValueError:
    #                 aux_level = 1
    #                 absolute_sequence[j] = (aux_level,phrase)
    #         #elif level == self.ROOT_LABEL:
    #             #absolute_sequence[j] = (1, phrase)
    #             current_level=aux_level
    #         else:                
    #             current_level+= level
    #             absolute_sequence[j] = (current_level,phrase)
    #     return absolute_sequence

    
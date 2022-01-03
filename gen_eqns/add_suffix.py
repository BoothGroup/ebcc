import re

def add_H_suffix_blocks(einsum_str):
    ''' Function to take the einsum output for a set of contractions, and for every
        instance of the interaction kernel or Fock matrix, change it so that it
        includes the attribute to point at the correct occ/virt block of those
        quantities.

        Limitations:
            Fock matrices must be labelled with a 'F'
            Interaction kernel must be labelled with an 'I'
            Electon-boson coupling tensor must be labelled with a 'g'
            Must only be max one F, I or g term in the contraction
            Quantity you are computing, as well as other matrices in the contraction cannot have an 'F', 'I', or 'g' in it
            Occupied indices i -> o
            Virtual indices a -> f
            Boson indices capital
    '''
    new_str = ''
    for f in einsum_str.splitlines():
        #print(f)
        result = re.search("'(.*)->", f)
        contractions = result.group(1).split(",")
        # These are a list of the contraction indices of each tensor
        #print(contractions)

        t = re.search("'(.*)'", f)
        # These are a list of the tensors
        tensors = re.split(', |\)',f[t.end():])[1:-1]
        #print(tensors)
        assert(len(contractions)==len(tensors))
        found_ham = False
        for i in range(len(contractions)):
            if tensors[i] == 'I':
                if found_ham:
                    # Should only find one term to replace, otherwise
                    # we need to make sure we replace the right instance
                    # in the original string
                    raise Exception
                found_ham = True
                #print('found I',i)
                # Have a look at the contraction string.
                assert(len(contractions[i])==4)
                suffix = '.'
                for char in contractions[i]:
                    if char in ['a', 'b', 'c', 'd', 'e', 'f']:
                        suffix += 'v'
                    elif char in ['i', 'j', 'k', 'l', 'm', 'n', 'o']:
                        suffix += 'o'
                    else:
                        raise Exception
                g = f.replace(' I',' I'+suffix)
            elif tensors[i] == 'g':
                if found_ham:
                    # Should only find one term to replace
                    raise Exception
                found_ham = True
                assert(len(contractions[i])==3)
                suffix = '.'
                for char in contractions[i]:
                    if char in ['a', 'b', 'c', 'd', 'e', 'f']:
                        suffix += 'v'
                    elif char in ['i', 'j', 'k', 'l', 'm', 'n', 'o']:
                        suffix += 'o'
                    elif not char.isupper():
                        # We can have capital letters denoting boson indices.
                        raise Exception
                g = f.replace(' g',' g'+suffix)
            elif tensors[i] == 'g_boscre':
                if found_ham:
                    # Should only find one term to replace
                    raise Exception
                found_ham = True
                assert(len(contractions[i])==3)
                suffix = '.'
                for char in contractions[i]:
                    if char in ['a', 'b', 'c', 'd', 'e', 'f']:
                        suffix += 'v'
                    elif char in ['i', 'j', 'k', 'l', 'm', 'n', 'o']:
                        suffix += 'o'
                    elif not char.isupper():
                        # We can have capital letters denoting boson indices.
                        raise Exception
                g = f.replace(' g_boscre',' g_boscre'+suffix)
            elif tensors[i] == 'F':
                #print('found F',i)
                if found_ham:
                    # Should only find one term to replace
                    raise Exception
                found_ham = True
                assert(len(contractions[i])==2)
                suffix = '.'
                for char in contractions[i]:
                    if char in ['a', 'b', 'c', 'd', 'e', 'f']:
                        suffix += 'v'
                    elif char in ['i', 'j', 'k', 'l', 'm', 'n', 'o']:
                        suffix += 'o'
                    else:
                        raise Exception
                # Replace the term in the original string
                g = f.replace(' F',' F'+suffix)
        if not found_ham:
            g = f
        #print(g)
        new_str = new_str + g + '\n'
    #print(new_str)
    return new_str

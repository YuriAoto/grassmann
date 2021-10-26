# cython: profile=True
"""Carry out the cluster decomposition




"""
import os

import numpy as np
import cython

from libc.stdlib cimport atoi

from util.variables import int_dtype


default_recipe_files = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../lib/recipes'))


cdef fill_recipes_from_ascii(int [:,:] recipes, recipes_f, int rank):
    """Read recipes from acii file"""
    cdef int i, k, k_index
    with open(f'{str(recipes_f)}_{rank}.dat', 'r') as f:
        i = 0
        for line in f:
            new_rule = list(map(lambda x: atoi(x.encode()), line.split()))
            for k_index, k in enumerate(new_rule):
                recipes[i, k_index] = k
            i += 1

cdef save_recipes_npy(recipes, recipes_f, rank):
    """Save recipe in a npy file"""
    np.save(recipes_f + '_{}.npy'.format(rank), np.array(recipes))

cdef load_recipes_npy(recipes, recipes_f, rank):
    """Load recipes in a npy file"""
    recipes = np.load(recipes_f + '_{}.npy'.format(rank))


def str_dec(decomposition):
    """Get a string version of the cluster decomposition"""
    d_str = ['=========================']
    for d in decomposition:
        d_str.append('-----')
        d_str.append(f'sign = {d[0]};')
        for det in d[1:]:
            d_str.append(str(det))
    d_str.append('=========================')
    return '\n'.join(d_str)


cdef class ClusterDecomposition:
    """Extension type to handle cluster decomposition"""


    def __cinit__(self, recipes_f=None):
        """Initialize: put all loaded_X to False"""
        self.loaded_D = False
        self.loaded_T = False
        self.loaded_Q = False
        self.loaded_5 = False
        self.loaded_6 = False
        self.loaded_7 = False
        self.loaded_8 = False

    def __init__(self, recipes_f=None):
        """Initialize: recipes file"""
        self.recipes_f = default_recipe_files if recipes_f is None else recipes_f

    cdef int[:,:] select_recipe(self, int rank):
        """Return a memview for the recipe associated to rank"""
        fill_in = False
        if rank == 2:
            if not self.loaded_D:
                self.recipes_D = np.zeros((3,8), dtype=np.dtype("i"))
                self.loaded_D = True
                fill_in = True
            recipes = self.recipes_D
        elif rank == 3:
            if not self.loaded_T:
                self.recipes_T = np.zeros((15,11), dtype=np.dtype("i"))
                self.loaded_T = True
                fill_in = True
            recipes = self.recipes_T
        elif rank == 4:
            if not self.loaded_Q:
                self.recipes_Q = np.zeros((114,14), dtype=np.dtype("i"))
                self.loaded_Q = True
                fill_in = True
            recipes = self.recipes_Q
        elif rank == 5:
            if not self.loaded_5:
                self.recipes_5 = np.zeros((1170,17), dtype=np.dtype("i"))
                self.loaded_5 = True
                fill_in = True
            recipes = self.recipes_5
        elif rank == 6:
            if not self.loaded_6:
                self.recipes_6 = np.zeros((15570,20), dtype=np.dtype("i"))
                self.loaded_6 = True
                fill_in = True
            recipes = self.recipes_6
        elif rank == 7:
            if not self.loaded_7:
                self.recipes_7 = np.zeros((256410,23), dtype=np.dtype("i"))
                self.loaded_7 = True
                fill_in = True
            recipes = self.recipes_7
        elif rank == 8:
            if not self.loaded_8:
                self.recipes_8 = np.zeros((5103000,26), dtype=np.dtype("i"))
                self.loaded_8 = True
                fill_in = True
            recipes = self.recipes_8
        else:
            raise ValueError('We can handle only up to octuples')
        if fill_in:
            fill_recipes_from_ascii(recipes, self.recipes_f, rank)
        return recipes

    cdef int n_rules(self, int rank):
        """Return the number of rules fir that rank"""
        if rank == 2:
            return 3
        if rank == 3:
            return 15
        if rank == 4:
            return 114
        if rank == 5:
            return 1170
        if rank == 6:
            return 15570
        if rank == 7:
            return 256410
        if rank == 8:
            return 5103000
        raise ValueError('We can handle only up to octuples')

    #@cython.boundscheck(False)  # Deactivate bounds checking
    #@cython.wraparound(False)   # Deactivate negative indexing
    cdef decompose(self, alpha_hp, beta_hp, mode='D'):
        """Carry out the decomposition
        
        Given the alpha and beta hole-particle indices,
        that indicate the excitation, return how this excitation
        is decomposed in clusters. This is returned as a list with the
        terms in the decomposition. Each of these terms is a tuple
        with the sign (+1 or -1) and all determinants that, multiplied,
        give this term.
        This decomposition is into doubles (mode = 'D')
        or into singles and doubles (mode = 'SD')
        
        Examples:
        ---------
        holes = [ijkl]
        particles = [abcd]
        return  [(sign, (i->a;j->b), (k->c;l->d)),
                 (sign, (i->a;k->b), (j->c;l->d)),
                    ...  ]
        
        holes = [ijklmn...]
        particles = [abcdef..]
        return  [(sign, (i->a;j->b), (k->c;l->d), (m->e;n->f), ...),
                 (sign, (i->a;k->b), (k->c;l->d), (m->e;n->f), ...),
                    ...   ]
        
        Parameters:
        -----------
        alpha_hp (2-tuple of np.array of int):
            alpha holes, alpha particles
        
        beta_hp (2-tuple of np.array of int):
            beta holes, beta particles
        
        mode (str: "D" or "SD")
            decompose into doubles or single and double
        
        Return:
        -------
        All terms in the decomposition that preserve spin projection,
        that is, do not change the number of alpha/beta electrons.
        These terms are returned as a list of lists, each starting with the
        sign of that decomposition, followd by tuples with (rank, alpha_hp, beta_hp)
        for that excitation.
        See the Example for the details
        
        """
        cdef int i, j, k, r
        cdef int first_h, first_p, n_alpha_h, n_alpha_p
        cdef int exc_rank, used_rank
        cdef int n_alpha = alpha_hp[0].shape[0]
        cdef int n_beta = beta_hp[0].shape[0]
        cdef int rank = n_alpha + n_beta
        cdef bint only_doubles = mode == 'D'
        cdef int [:,:] rules = self.select_recipe(rank)
        cdef int n_rules = self.n_rules(rank)
        all_decompositions = []
        # Example:
        # alpha_hp, beta_hp = ([i,j,k], [l]), ([a,b,c], [d])
        # orbitals = [i,j,k,l,a,b,c,d]
        #
        orbitals = np.concatenate((alpha_hp[0], beta_hp[0],
                                   alpha_hp[1], beta_hp[1]))
#        print(np.array(rules))
        for i in range(n_rules):
            if only_doubles and rank // 2 != rules[i, 1]:
                continue
            use_this_decomposition = True
            new_decomposition = [rules[i, 0]]
            j = 2
            used_rank = 0
 #           print(np.array(rules[i,:]))
            while used_rank < rank:
                exc_rank = rules[i, j]
                # I think this is not needed (for the version of the recipes with S and D):
                # if exc_rank > 2 or exc_rank == 1 and only_doubles:
                #     use_this_decomposition = False
                #     break
                first_h = j + 1
                first_p = first_h + exc_rank
                n_alpha_h = 0
                for k in range(first_h, first_p):
                    if rules[i, k] < n_alpha:
                        n_alpha_h += 1
                n_alpha_p = 0
                for k in range(first_p, first_p + exc_rank):
                    if rules[i, k] < rank + n_alpha:
                        n_alpha_p += 1
                if n_alpha_h != n_alpha_p:
                    use_this_decomposition = False
                    break
                alpha_hp = ([], [])
                beta_hp = ([], [])
                for r in range(exc_rank):
                    if r < n_alpha_h:
                        alpha_hp[0].append(orbitals[rules[i, first_h + r]])
                        alpha_hp[1].append(orbitals[rules[i, first_p + r]])
                    else:
                        beta_hp[0].append(orbitals[rules[i, first_h + r]])
                        beta_hp[1].append(orbitals[rules[i, first_p + r]])
                new_decomposition.append((exc_rank,
                                          (np.array(alpha_hp[0], dtype=int_dtype),
                                           np.array(alpha_hp[1], dtype=int_dtype)),
                                          (np.array(beta_hp[0], dtype=int_dtype),
                                           np.array(beta_hp[1], dtype=int_dtype))))
                j += 2 * exc_rank + 1
                used_rank += exc_rank
            if use_this_decomposition:
                all_decompositions.append(new_decomposition)
        return all_decompositions

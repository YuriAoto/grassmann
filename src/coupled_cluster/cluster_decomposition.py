"""


"""
import os

import numpy as np

from wave_functions.slater_det import get_slater_det_from_excitation


default_recipe_files = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../lib/recipes'))


def cluster_decompose(alpha_hp, beta_hp, ref_det, mode='D', recipes_f=None):
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
    return  [(sign, det_ij_ab, det_kl_cd),
             (sign, det_ik_ab, det_jl_cd),
                ...  ]
    
    holes = [ijklmn...]
    particles = [abcdef..]
    return  [(sign, det_ij_ab, det_kl_cd, det_mn_ef, ...),
             (sign, det_ik_ab, det_kl_cd, det_mn_ef, ...),
                ...   ]
    
    Parameters:
    -----------
    alpha_hp (2-tuple of np.array of int):
        alpha holes, alpha particles
    
    beta_hp (2-tuple of np.array of int):
        beta holes, beta particles
    
    ref_det (SlaterDet)
        The reference Slater determinant
    
    mode (str: "D" or "SD")
        decompose into doubles or single and double
    
    recipes_f (None or str with a file name prefix)
        use the files
            recipes_f + '_{}.dat'.format(rank)
        as the recipes to obtain the decomposition.
        If None, use the module variable default_recipe_files
    
    Return:
    -------
    All terms in the decomposition that preserve spin projection,
    that is, do not change the number of alpha/beta electrons.
    See the Example for the details
    
    """
    recipes_f = default_recipe_files if recipes_f is None else recipes_f
    n_alpha = alpha_hp[0].shape[0]
    n_beta = beta_hp[0].shape[0]
    rank = n_alpha + n_beta
    all_decompositions = []
    only_doubles = mode == 'D'
    # Example:
    # alpha_hp, beta_hp = ([i,j,k], [l]), ([a,b,c], [d])
    # orbitals = [i,j,k,l,a,b,c,d]
    #
    orbitals = np.concatenate((alpha_hp[0], beta_hp[0],
                               alpha_hp[1], beta_hp[1]))
    with open(recipes_f + '_{}.dat'.format(rank), 'r') as f:
        for line in f:
            lspl = list(map(int, line.split()))
            if only_doubles and rank // 2 != lspl[1]:
                continue
            use_this_decomposition = True
            new_decomposition = [lspl[0]]
            i = 2
            while i < len(lspl):
                exc_rank = lspl[i]
                if exc_rank > 2 or exc_rank == 1 and only_doubles:
                    use_this_decomposition = False
                    break
                first_h = i + 1
                first_p = first_h + exc_rank
                n_alpha_h = sum(orbind < n_alpha
                                for orbind in lspl[first_h:first_p])
                n_alpha_p = sum(orbind < rank + n_alpha
                                for orbind in lspl[first_p:first_p+exc_rank])
                if n_alpha_h != n_alpha_p:
                    use_this_decomposition = False
                    break
                alpha_hp = ([], [])
                beta_hp = ([], [])
                for r in range(exc_rank):
                    if r < n_alpha_h:
                        alpha_hp[0].append(orbitals[lspl[first_h + r]])
                        alpha_hp[1].append(orbitals[lspl[first_p + r]])
                    else:
                        beta_hp[0].append(orbitals[lspl[first_h + r]])
                        beta_hp[1].append(orbitals[lspl[first_p + r]])
                new_decomposition.append(get_slater_det_from_excitation(
                    ref_det, 0.0, alpha_hp, beta_hp))
                i += 2 * exc_rank + 1
            if use_this_decomposition:
                all_decompositions.append(new_decomposition)
    return all_decompositions

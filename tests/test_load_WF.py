import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath('__file__')) + '/../')

import numpy as np

import dGr_util as util
import dGr_WF_int_norm as IntN
import dGr_FCI_Molpro as FCI
import dGr_general_WF as genWF

Ua, Ub = FCI.transf_orb_from_to('Li2_min_basis_R_5.0/RHF.out',
                                'Li2_min_basis_R_5.0/UHF.out')


if sys.argv[1].lower() != 'fci':
#    print('Using interm norm:')
    wf = IntN.Wave_Function_Int_Norm.from_Molpro('Li2_min_basis_R_5.0/CISD_wf.out')
    Ua = Ua[:,:wf.n_alpha]
    Ub = Ub[:,:wf.n_beta]
    wf.calc_norm()
    for det in wf.all_dets():
        if abs(det.c) > 1.0E-12:
            if isinstance(det, genWF.Doubly_Exc_Det):
                l = [det.c]
                if det.spin_ia * det.spin_jb < 0:
                    Ia = util.get_I(wf.n_alpha, det.i, det.a)
                    Ib = util.get_I(wf.n_beta,  det.j, det.b)
                elif det.spin_ia > 0:
                    Ia = util.get_I(wf.n_alpha,
                               [det.i, det.j],
                               sorted([det.a, det.b]))
                    Ib = util.get_I(wf.n_beta)
                else:
                    Ia = util.get_I(wf.n_alpha)
                    Ib = util.get_I(wf.n_beta,
                               [det.i, det.j],
                               sorted([det.a, det.b]))
                l.extend(map(lambda x: x+1, Ia + Ib))
                print (l)
else:
#    print('Using FCI:')
    wf = FCI.Molpro_FCI_Wave_Function('Li2_min_basis_R_5.0/CISD_wf.out',
                                      FCI_file_name='Li2_min_basis_R_5.0/FCI_templ.out')
    Ua = Ua[:,:wf.n_alpha]
    Ub = Ub[:,:wf.n_beta]
    # print('Reference:')
    # for det in wf.determinants:
    #     if abs(det[0]) > 1.0E-12 and FCI.rank_of_exc(det) == 0:
    #         print(det)
    # print('Singles:')
    # for det in wf.determinants:
    #     if abs(det[0]) > 1.0E-12 and FCI.rank_of_exc(det) == 1:
    #         print(det)
    # print('Doubles:')
    for det in wf.determinants:
        if abs(det[0]) > 1.0E-12 and FCI.rank_of_exc(det) == 2:
            print(det)
    # print('Triples and higher:')
    # for det in wf.determinants:
    #     if abs(det[0]) > 1.0E-12 and FCI.rank_of_exc(det) > 2:
    #         print(det)



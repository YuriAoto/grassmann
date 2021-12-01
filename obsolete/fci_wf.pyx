cdef OrbitalSpace orbspace_from_Molpro_line(str l,
                                            FullOrbitalSpace orbspace,
                                            double Ms,
                                            str molpro_output,
                                            int line_number)


cdef OrbitalSpace orbspace_from_Molpro_line(str l,
                                            FullOrbitalSpace orbspace,
                                            double Ms,
                                            str molpro_output,
                                            int line_number):
    """Give the orbital occupation of a FCI line Molpro output 
    
    Parameters:
    -----------
    l (str)
        The line with a configuration, from the FCI program in Molpro
        to be converted to a Slater Determinant.
    
    orbspace (FullOrbitalSpace)
        All orbital spaces
        
    Ms (float)
        Ms of total wave function (n_alpha - n_beta)/2
    
    molpro_output (str, optional, default='')
        The output file name (only for error message)
    
    line_number (int, optional, default=-1)
        The line number in Molpro output
    
    Returns:
    --------
    A OrbitalSpace, see the examples
    
    Raises:
    molpro.MolproInputError
    
    Examples:
    ---------
    
    # if n_irrep = 4, orbspace.full = (6,2,2,0), orbspace.froz = (0,0,0,0) then
    
    -0.162676901257  1  2  7  1  2  7
    gives
    [2 1 0 0 2 1 0 0] (type='F')
    
    -0.049624632911  1  2  9  1  2  7
    gives
    [(0,1,3) () () () (0,1,5) () () ()]
    
    0.000000000000  1  2  9  1  2 10
    gives
    [(0,1) () (0) () (0,1) () (1) ()]
    
    # but if orbspace.froz = (1,1,0,0) then the above cases give
         (because frozen electrons are indexed first in Molpro convention)
    
    [(0,5) (0) () () (0,5) (0) () ()]
    
    [(0,2) (0) () () (0,4) (0) () ()]
    
    [(0) (0) (0) () (0) (0) (1) ()]


    """
    cdef int irrep, i, orb
    cdef OrbitalSpace orbsp
    lspl = l.split()
    final_occ = [list(range(orbspace.froz[irp]))
                 for irp in range(2 * orbspace.n_irrep)]
    n_tot_frozen = sum(map(len, final_occ)) // 2
    try:
        coeff = float(lspl[0])
        occ = [int(x) - 1 for x in lspl[1:] if int(x) > n_tot_frozen]
    except Exception as e:
        raise molpro.MolproInputError(
            "Error when reading FCI configuration. Exception was:\n"
            + str(e),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    if len(occ) + 2 * n_tot_frozen + 1 != len(lspl):
        raise molpro.MolproInputError(
            "Inconsistency in number of frozen orbitals for FCI:\n"
            + str(orbspace.froz),
            line=l,
            line_number=line_number,
            file_name=molpro_output)
    total_orbs = [sum(orbspace.froz[irp] for irp in range(orbspace.n_irrep))]
    for i in range(orbspace.n_irrep):
        total_orbs.append(total_orbs[-1] + orbspace.full[i] - orbspace.froz[i])
    irrep = irrep_shift = 0
    ini_beta = (len(occ) + int(2 * Ms)) // 2
    for i, orb in enumerate(occ):
        if i == ini_beta:
            irrep_shift = orbspace.n_irrep
            irrep = 0
        while True:
            if irrep == orbspace.n_irrep:
                raise molpro.MolproInputError(
                    'Configuration is not consistent with dimension'
                    ' of full orbital space: ' + str(orbspace.full),
                    line=l,
                    line_number=line_number,
                    file_name=molpro_output)
            if total_orbs[irrep] <= orb < total_orbs[irrep + 1]:
                final_occ[irrep + irrep_shift].append(orb - total_orbs[irrep]
                                                      + orbspace.froz[irrep])
                break
            else:
                irrep += 1
    for i, o in enumerate(final_occ):
        final_occ[i] = np.array(o, dtype=int_dtype)
    orbsp = OrbitalSpace(dim=list(map(len, final_occ)))
    return orbsp





self.orbspace.set_ref(orbspace_from_Molpro_line(
    line, self.orbspace, self.Ms, f_name, line_number))



@tests.category('SHORT')
class OrbspFromMolproTestCase(unittest.TestCase):

    def setUp(self):
        self.orbspace = FullOrbitalSpace(n_irrep=4)
        self.orbspace.set_full(OrbitalSpace(dim=[6, 2, 2, 0], orb_type='R'))

    def test1(self):
        orbsp = orbspace_from_Molpro_line('-0.162676901257  1  2  7  1  2  7',
                                          self.orbspace, 0.0, '', -1)
        self.assertEqual(orbsp, OrbitalSpace(dim=[2,1,0,0, 2,1,0,0], orb_type='F'))

    def test2(self):
        orbsp = orbspace_from_Molpro_line('-0.162676901257  1  2  7  1  2  9',
                                          self.orbspace, 0.0, '', -1)
        self.assertEqual(orbsp, OrbitalSpace(dim=[2,1,0,0, 2,0,1,0], orb_type='F'))

    def test3(self):
        orbsp = orbspace_from_Molpro_line('-0.162676901257  1  2  9  1  2  7',
                                          self.orbspace, 0.0, '', -1)
        self.assertEqual(orbsp, OrbitalSpace(dim=[2,0,1,0, 2,1,0,0], orb_type='F'))

    def test4(self):
        orbsp = orbspace_from_Molpro_line('-0.162676901257  1  2  9  1  2  9',
                                          self.orbspace, 0.0, '', -1)
        self.assertEqual(orbsp, OrbitalSpace(dim=[2,0,1,0, 2,0,1,0], orb_type='F'))

    def test5(self):
        orbsp = orbspace_from_Molpro_line('-0.162676901257  1  7  8  1  2  9',
                                          self.orbspace, 0.0, '', -1)
        self.assertEqual(orbsp, OrbitalSpace(dim=[1,2,0,0, 2,0,1,0], orb_type='F'))

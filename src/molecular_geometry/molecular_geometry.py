"""Atoms and molecular geometry.

Yuri Aoto

Start - Jun 2018
Add interface to Knizia's WMME - Jul 2018
Forked to separate integrals from Molecular geometry - Feb 2019
Added to grassmann - Aug 2020
"""
import re
import math

import integrals
from util.variables import ANG_to_a0
from molecular_geometry.periodic_table import ATOMS

re_get_number = re.compile(r'\d+$')


def norm_sq(x):
    """Return sum_i x[i]**2."""
    S = 0.0
    for i in x:
        S += i**2
    return S


class Atom():
    """An atom in space.

    Contains all information about a particular atom
    in the molecular geometry:

    element : the chemical element
    coord : Its coordinates in space
    atomic_number : Its atomic number
    label : An (optional) unique label
    basis : The basis set of this atom

    """

    def __init__(self, element, coord, label=None):
        """Initialize the Atom.

        Parameters
        ----------
        element : string
            the atomic element
        coord : list of len 3
            the x,y,z coordinates of the atom
        label : string, optional
            a label, to uniquely specify the atom
        """
        self.element = element
        self.coord = coord
        try:
            self.atomic_number = ATOMS.index(element)
        except Exception:
            raise Exception('Problem when getting atomic number of ' +
                            element)
        if self.atomic_number == 0:
            self.atomic_number = 1.0
        if label is None:
            self.label = element
        else:
            self.label = label
        self.basis = None

    def __repr__(self):
        """Return an representation of Atom."""
        x = '<Atom: ' + self.label + ': ' + self.element + '; ' +\
            repr(self.coord)
        if self.basis is None:
            x += '>'
        else:
            x += '; basis = ' + repr(self.basis) + '>'
        return x

    def __str__(self):
        """Return a str of Atom."""
        x = 'Atom: ' + self.element + '\n'
        x = 'Atomic number: ' + str(self.atomic_number) + '\n'
        x += 'Coordinates: ' + str(self.coord) + '\n'
        x += 'Label: ' + self.label + '\n'
        if self.basis is None:
            pass
        else:
            x += 'Basis = ' + str(self.basis)
        return x

    def coord_rel_to(self, other):
        """Return the cartesian coordinates relative to other."""
        return [self.coord[0] - other.coord[0],
                self.coord[1] - other.coord[1],
                self.coord[2] - other.coord[2]]

    def distance_to(self, other):
        """Return the distance to the atom other."""
        x = 0.0
        coord_rel = self.coord_rel_to(other)
        for c in coord_rel:
            x += c**2
        return math.sqrt(x)


class MolecularGeometry():
    """A set of atoms in space.
    
    
    """
    
    def __init__(self):
        """Initialise the molecular geometry.
        
        Parameters
        ----------
        geom_file (string, optional, default=None)
            If not None, the geometry is read from this file,
            in the standard .xyz format
        """
        self.name = ''
        self._atoms = []
        self.atomic_basis_set = None
        self.integrals = None
        self.charge = 0
        self._nucl_rep = None
        self._n_elec = None

    def __iter__(self):
        return iter(self._atoms)

    def __getitem__(self, key):
        """Return atom by position or label."""
        if isinstance(key, int):
            return self._atoms[key]
        elif isinstance(key, str):
            for at in self._atoms:
                if at.label == key:
                    return at
        raise ValueError

    def __repr__(self):
        """Return the repr of the MolecularGeometry."""
        x = '<MolecularGeometry: '
        x += ';'.join(map(repr, self._atoms))
        x += '>'
        return x

    def __str__(self):
        """Return the str of MolecularGeometry."""
        x = ['Molecule'
             + (' (' + self.name + ')' if self.name else '')
             + ':']
        for at in self._atoms:
            x.append(at.label + '    '
                     + '   '.join(map(str, at.coord)))
        return '\n'.join(x)

    def __len__(self):
        """The number of atoms"""
        return len(self._atoms)

    @classmethod
    def from_xyz_file(cls, file_name):
        """Construct a MolecularGeometry from a xyz file"""
        new_geom = cls()
        with open(file_name, 'r') as f:
            int(f.readline())  # n atoms
            new_geom.name = f.readline()
            for line in f:
                try:
                    comment_pos = line.index('#')
                except ValueError:
                    pass
                else:
                    line = line[0:comment_pos]
                line = line.split()
                if len(line) == 0:
                    continue
                element = re_get_number.sub('', line[0])
                new_geom.add_atom(Atom(element, tuple(map(
                    lambda x: float(x) * ANG_to_a0, line[1:])),
                                       line[0]))
        return new_geom

    def add_atom(self, atom):
        """Add a new Atom to the Molecular Geometry."""
        self._n_elec = None
        self._nucl_rep = None
        self._atoms.append(atom)

    @property
    def n_elec(self):
        """Return the number of electrons in the MolecularGeometry."""
        if self._n_elec is None:
            self._n_elec = 0
            for at in self._atoms:
                self._n_elec += (at.atomic_number
                                 if isinstance(at.atomic_number, int) else
                                 1)
            self._n_elec -= self.charge
        return self._n_elec

    @property
    def nucl_rep(self):
        """Return the nuclear repulsion term."""
        if self._nucl_rep is None:
            self._nucl_rep = 0.0
            for i1, at1 in enumerate(self._atoms):
                for i2, at2 in enumerate(self._atoms):
                    if i2 <= i1:
                        continue
                    self._nucl_rep += (
                        at1.atomic_number * at2.atomic_number /
                        math.sqrt(norm_sq(at1.coord_rel_to(at2))))
        return self._nucl_rep

    def calculate_integrals(self, basis_set, int_meth=None):
        """Calculates the integrals.
        
        Parameters
        ----------
        basis_set : string
            Contains the basis set information. See class Atomic_Basis
        
        int_meth : string, optional, default = None
            The method to calculate the integrals
        """
        self.atomic_basis_set = basis_set
        self.integrals = integrals.Integrals(self, basis_set,
                                             method=int_meth)

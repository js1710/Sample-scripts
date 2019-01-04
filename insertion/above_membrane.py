from __future__ import print_function, division
import numpy as np
import MDAnalysis as mda
import argparse
import poission_disk


def insert_molecule(molecule, system, memb_sel, dist, rand_pos, leaflet="upper", rand=True):
    '''
    Inserts a molecule above or below a membrane
    :param molecule: mdanalysis inverse object for molecule to be added
    :param system: mdanalysis universe object
    :param memb_sel: mdanalysis style selection string for membrane
    :param dist: distance in nm to insert above membrane
    :param rand_pos: points in the xy plane to insert molecule
    :param leaflet: the leaflet to insert the molecule with respect to
    :param rand: If true molecule is inserted in the xy plane using 'rand_pos' otherwise the center of the simulation
                 box is used
    :return: A new mdanalysis object containing the inserted molecule
    '''

    membrane = system.select_atoms(memb_sel)
    num = molecule.atoms.n_atoms

    # center protein to membrane#
    cog1 = membrane.center_of_geometry(pbc=False)
    cog2 = molecule.atoms.center_of_geometry(pbc=False)
    move = cog1 - cog2
    molecule.atoms.translate(move)

    # move to random xy position
    if rand:
        cog2 = molecule.atoms.center_of_geometry(pbc=False)[:2]
        move = rand_pos - cog2
        molecule.atoms.translate(np.append(move, 0.))

    new_system = mda.Merge(molecule.atoms, system.atoms)
    new_system.dimensions = system.dimensions
    molecule = new_system.atoms[:num]
    membrane = new_system.select_atoms(memb_sel)

    if leaflet == "upper":
        #insert molecule above membrane
        min_ind = np.argmin(molecule.positions[:, 2])
        min_atom = molecule.atoms[min_ind]
        min_z = min_atom.position[2]
        membrane = system.select_atoms("{0}".format(memb_sel))

        max_z = np.max(membrane.positions[:, 2])
        diffz = min_z - max_z
        movez = dist * 10. - diffz

    else:
        #insert molecule below membrane
        molecule.atoms.rotateby(180., [1, 0, 0])
        max_ind = np.argmax(molecule.positions[:, 2])
        max_atom = molecule.atoms[max_ind]
        max_z = max_atom.position[2]
        membrane = system.select_atoms("{0}".format(memb_sel))

        min_z = np.min(membrane.positions[:, 2])
        diffz = max_z - min_z
        movez = -dist * 10. - diffz

    molecule.translate([0, 0, movez])
    #set size of new box in the z dimension
    protein_maxz = np.max(molecule.positions[:, 2])
    if protein_maxz > new_system.dimensions[2]:
        new_system.dimensions[2] = protein_maxz + 10.

    return new_system





if __name__ == '__main__':
    #commandline parser options
    parser = argparse.ArgumentParser("Insert molecules above or below the membrane and deletes overlapping residues",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("protein", help="protein structure file")
    parser.add_argument("system", help="membrane structure file")
    parser.add_argument("-ms", "--memb_sel", default="all", help="membrane mdanalysis selection string")
    parser.add_argument("-d", "--dist", type=float, default=1.0, help="distance above membrane in nm to insert molecules")
    parser.add_argument("-o", "--output", default="merged_system.gro", help="name and format of output file")
    parser.add_argument("-n", "--n_add", type=int, default=1, help="number of molcules to add per leaflet")
    parser.add_argument("-l", "--leaflets", default="both", choices=["upper", "lower", "both"], help="side "
                                                                            "of leaflet to add molecule.")
    parser.add_argument("--norand", action="store_true", help="Use to turn off random molecule placement and "
                                                              "instead insert the molecule in the center of the system.")
    parser.add_argument("-r", "--remove", type=float, default=0., help="Remove any residues within a certain "
                                                                             "distance of the inserted molecules")
    parser.add_argument("--closest", type=float, default=3., help="Center of geometry of inserted molecules will not be"
                                                                  "closer than this value")
    args = parser.parse_args()

    protein = mda.Universe(args.protein)
    rand = False if args.norand else True

    #load system and generate random 2D points within the simulation box in the xy plane#
    system = mda.Universe(args.system)
    pdisk = poission_disk.PoissonDisc(system.dimensions[:2], 30) #points will not be closer than 3 nm
    pdisk.sample()
    if args.leaflets == "both":
        #select random 2D poisitions to insert molecules at
        rand_positions = pdisk.samples[np.random.choice(pdisk.samples.shape[0], args.n_add * 2, replace=False)]
        for i, rand_pos in enumerate(rand_positions):
            if i % 2 == 0:
                system = insert_molecule(protein, system, args.memb_sel, args.dist, rand_pos, leaflet="upper", rand=rand)
            else:
                system = insert_molecule(protein, system, args.memb_sel, args.dist, rand_pos, leaflet="lower", rand=rand)

    else:
        rand_positions = pdisk.samples[np.random.choice(pdisk.samples.shape[0], args.n_add, replace=False)]
        for i, rand_pos in enumerate(rand_positions):
            system = insert_molecule(protein, system, args.memb_sel, args.dist, rand_pos, leaflet=args.leaflets, rand=rand)

    if args.remove > 0:
        #remove atoms within a cutoff of inserted molecules
        molecule_selection = "resname " + " ".join(protein.residues.resnames)
        system = system.atoms.select_atoms("{0} or not byres around {1} {0}".format(molecule_selection, args.remove))
    system.atoms.write(args.output)





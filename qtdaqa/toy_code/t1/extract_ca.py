from pathlib import Path

class ProteinParser:
    def __init__(self, inputfile):
        self.inputfile = inputfile

    def extract_ca_atoms_info(self):
        from Bio import PDB
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", self.inputfile)

        ca_atoms_info = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == " " and residue.has_id("CA"):
                        atom = residue["CA"]
                        chain_id = chain.id
                        res_id = residue.get_id()[1]
                        ins_code = residue.get_id()[2].strip()
                        coords = atom.get_coord()
                        res_name = residue.get_resname()
                        print(chain.id,res_id,res_name)
                        ca_atoms_info.append((chain_id, res_id, res_name, ins_code, coords))

        return ca_atoms_info

# Example usage
parser = ProteinParser("example.pdb")
info = parser.extract_ca_atoms_info()
print(info)

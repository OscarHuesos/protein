# protein
C++ interface to read, preprocessing and prepare the data to be trained of the .pdb files. 
This code also calculates the ASA CUDA calculation, the interface residues using CUDA, and b-factor
normalization.

The processing of each .pdb file is not automatic. The executable will be created at bin folder, then each
.pdb is placed manually at bin folder (1es7FH.pdb as  example here).

We select the .pdb file to be read at main.cc

To copile its necessary Cmake minimum required version: 3.5.1 , gcc  5.4.0  and CUDA 8.0.

run on the protein-main folder path:

cmake .
make

Then, go to bin and run ./protein

The output will generate four files:

-The data_generated.txt is the list format of the relevant information used to trained (not labelled yet).
-datap.y is a python file in format of the list of residues to be proccessed by PyQuante2 (energies calculation) in queue.
-datasa.txt generate the ASA of each residue individually (considered at data_generated.txt file).

NOTE:
If available, the conservation score and energies are added in a sigle step at data_generated.txt (added in bin folder).


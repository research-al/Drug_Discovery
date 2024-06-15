import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_ecfp8(file_path):
  # Read the data file into a pandas DataFrame
  print('fingerprint.py running\n')
  df = pd.read_csv(file_path)

  affinity = df['affinity']

  # Initialize an empty list to store the fingerprints
  fingerprints = []

  # Iterate through each SMILES string in the 'SMILES' column
  for smiles in df['SMILES']:
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # Check if RDKit successfully converted the SMILES to a molecule
      # Generate ECFP fingerprint with radius 4 (ECFP8) and 1024 bits
      fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
      # Convert the fingerprint to a numpy array
      fp_array = np.zeros((0, ), dtype=np.int8)
      AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)

      # Append the fingerprint array to the list of fingerprints
      fingerprints.append(fp_array)

  # Convert the list of fingerprints to a numpy array
  fingerprints_array = np.array(fingerprints)
  return fingerprints_array, affinity


'''
file_path= 'ActiveLearning_BindingAffinity/TYK2_final.csv'
fingerprints_array, affinity = smiles_to_ecfp8(file_path)
print(f"fingerprints_array: {fingerprints_array}")
print(fingerprints_array.shape) 
'''

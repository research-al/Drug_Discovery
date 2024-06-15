from activelearning import train_gp_regression
from fingerprint import smiles_to_ecfp8


def main():
  file_path = 'ActiveLearning_BindingAffinity/TYK2_final.csv'
  fingerprints_array, affinity = smiles_to_ecfp8(file_path)
  training_losses = train_gp_regression(fingerprints_array, affinity)
  print('main.py running')


if __name__ == "__main__":
  main()

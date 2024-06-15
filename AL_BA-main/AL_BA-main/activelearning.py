import gpytorch
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from sklearn.model_selection import KFold, train_test_split


def train_gp_regression(fingerprints_array, affinity_values):
  print("activelearning.py running\n")
  # Convert features and target values to PyTorch tensors
  train_x = torch.tensor(fingerprints_array, dtype=torch.float32)
  train_y = torch.tensor(affinity_values, dtype=torch.float32)

  # Split the data into training and testing sets
  train_x, test_x, train_y, test_y = train_test_split(train_x,
                                                      train_y,
                                                      test_size=0.8,
                                                      random_state=42)

  # Set up k-fold cross-validation
  kf = KFold(n_splits=5, shuffle=True)

  class GPRegressionModel(ExactGP):

    def __init__(self, train_x, train_y, likelihood):
      super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
      self.mean_module = ConstantMean()
      self.covar_module = RBFKernel()

    def forward(self, x):
      mean_x = self.mean_module(x)
      covar_x = self.covar_module(x)
      return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

  # Initialize the likelihood
  likelihood = GaussianLikelihood()

  all_training_losses = []
  for fold, (train_index, val_index) in enumerate(kf.split(train_x)):
    print(f"Fold: {fold+1}")

    # Get the training and validation data for this fold
    fold_train_x, fold_val_x = train_x[train_index], train_x[val_index]
    fold_train_y, fold_val_y = train_y[train_index], train_y[val_index]

    # Initialize the model
    model = GPRegressionModel(fold_train_x, fold_train_y, likelihood)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Create the MLL object
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    model.train()
    likelihood.train()
    training_losses = []
    epoch = 50
    for i in range(epoch):
      optimizer.zero_grad()
      output = model(fold_train_x)
      loss = -mll(output, fold_train_y)
      loss.backward()
      print('Iter %d/%d || Loss: %.3f || noise: %.3f' %
            (i + 1, epoch, loss.item(), model.likelihood.noise.item()))

      optimizer.step()
      training_losses.append(loss.item())

    all_training_losses.append(training_losses)

    # Evaluate on the validation set for this fold
    model.eval()
    likelihood.eval()
    with torch.no_grad():
      output_val = model(fold_val_x)
      val_loss = -mll(output_val, fold_val_y).item()

  return all_training_losses

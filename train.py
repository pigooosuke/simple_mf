import argparse
from dataset import Dataset
from model import MFModel
from evaluate import evaluate

import numpy as np


def main():
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='data/ml-1m',
                      help='Path to the dataset')
  parser.add_argument('--epochs', type=int, default=128,
                      help='Number of training epochs')
  parser.add_argument('--embedding_dim', type=int, default=8,
                      help='Embedding dimensions, the first dimension will be '
                           'used for the bias.')
  parser.add_argument('--regularization', type=float, default=0.0,
                      help='L2 regularization for user and item embeddings.')
  parser.add_argument('--negatives', type=int, default=8,
                      help='Number of random negatives per positive examples.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='SGD step size.')
  parser.add_argument('--stddev', type=float, default=0.1,
                      help='Standard deviation for initialization.')
  args = parser.parse_args()

  # Load the dataset
  dataset = Dataset(args.data)
  train_pos_pairs = np.column_stack(dataset.trainMatrix.nonzero())
  test_ratings, test_negatives = (dataset.testRatings, dataset.testNegatives)
  print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' % (
      dataset.num_users, dataset.num_items, train_pos_pairs.shape[0],
      len(test_ratings)))

  # Initialize the model
  model = MFModel(dataset.num_users, dataset.num_items,
                  args.embedding_dim-1, args.regularization, args.stddev)

  # Train and evaluate model
  hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
  print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
        % (0, hr, ndcg))
  for epoch in range(args.epochs):
    # Training
    _ = model.fit(train_pos_pairs, learning_rate=args.learning_rate,
                  num_negatives=args.negatives)

    # Evaluation
    hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
    print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
          % (epoch+1, hr, ndcg))


if __name__ == '__main__':
  main()

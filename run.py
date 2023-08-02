import nanno.data_loader as dl
from nanno.core import ANN

training_set, evaluation_set = dl.get_two_by_two()

sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
n_nodes = [n_pixels, n_pixels]

autoencoder = ANN(layers=None)

if __name__ == "__main__": 
    autoencoder.train(training_set)
    autoencoder.evaluate(evaluation_set)
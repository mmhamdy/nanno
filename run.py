import nanno.data_loader as dl

training_set, evaluation_set = dl.get_two_by_two()

sample = next(training_set())
n_pixels = sample.shape[0] * sample.shape[1]
n_nodes = [n_pixels, n_pixels]


if __name__ == "__main__": 
    print(n_pixels)
    print(n_nodes)
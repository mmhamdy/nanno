import numpy as np

def get_two_by_two():
    examples = [
        np.array([
            [0,0],
            [1,1]
        ]),
        np.array([
            [1,0],
            [1,1]
        ]),
        np.array([
            [0,0],
            [0,1]
        ]),
        np.array([
            [0,1],
            [1,0]
        ]),
        np.array([
            [1,0],
            [0,1]
        ]),
        np.array([
            [1,1],
            [0,0]
        ]),
        np.array([
            [0,1],
            [1,1]
        ]),
        np.array([
            [0,1],
            [0,0]
        ]),
        np.array([
            [1,0],
            [1,0]
        ]),
        np.array([
            [0,1],
            [0,1]
        ]),
        np.array([
            [1,0],
            [0,0]
        ]),
        np.array([
            [1,1],
            [0,1]
        ]),
    ]

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]
    
    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    return training_set, evaluation_set
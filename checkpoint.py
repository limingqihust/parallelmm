import pickle
import os
import numpy as np

class CheckpointManager():
    def __init__(self):
        self.checkpoint = {}
    
    def save_checkpoint(self, id, C, i, j):
        checkpoint = (np.copy(C), i, j)
        filename = f"checkpoint_{id}.pkl"
        with open(filename, "wb") as file:
            pickle.dump(checkpoint, file)

    def load_checkpoint(self, id):
        filename = f"checkpoint_{id}.pkl"
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as file:
            checkpoint = pickle.load(file)
        return checkpoint
        
from charm4py import charm, Chare, Array, Reducer
import numpy as np
from config import Config
import time

class MatrixGenerator:
    def __init__(self, call) -> None:
        self.generator = call
        
    def __call__(self, row, col):
        return self.generator(row, col)

class MatrixMul(Chare):

    def set_data(self, A_slice, B, main_proxy, index, fail_index, struggle_time):
        self.A_slice = A_slice
        self.B = B
        self.main_proxy = main_proxy
        self.index = index
        self.fail_index = fail_index
        self.struggle_time = struggle_time

    def multiply(self):
        start = time.time()
        if self.index == self.fail_index:
            time.sleep(self.struggle_time*10e-3)
        C_slice = np.dot(self.A_slice, self.B)
        end = time.time()
        
        comp_time = end - start
        
        self.reduce(self.main_proxy.gather_results, (C_slice, self.index, comp_time), Reducer.gather)

class Encoder:
    
    def __call__(self, shards):
        return np.sum(shards, axis=0)

class Decoder:
    
    def __call__(self, shards):
        parity = shards[-1]
        for shard in shards[:-1]:
            parity -= shard
        return parity

class Main(Chare):

    def start(self, n, A, B, type, fail_index, struggle_time):
        # Split A into sub-matrices
        self.type = type
        self.n = n
        A_slices = np.array_split(A, n-1, axis=0)

        enc = Encoder()
        enc_slice = enc(A_slices)
        A_slices.append(enc_slice)
        
        # Create an Array of MatrixMul objects
        self.workers = Array(MatrixMul, n)

        # Set data for each worker and start the multiplication
        for i in range(n):
            self.workers[i].set_data(A_slices[i], B, self.thisProxy, i, fail_index, struggle_time)
            self.workers[i].multiply()

    def gather_results(self, results):
        # Initialize a list to hold the partial results
        results_dict = {}
        indices = []
        comp_times = {}

        # Process results to map them by worker index
        for C_slice, index, comp_time in results:
            results_dict[index] = C_slice
            indices.append(index)
            comp_times[index] = comp_time
        
        decode_time = 0
        if self.type == "CDC":
            # choose the one with the largest comp_time, and discard it
            l_index = max(comp_times, key=lambda x: comp_times[x])
            del results_dict[l_index]
            del comp_times[l_index]
                
            # Sort and merge results based on worker index
            sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
            
            # if need to decode
            if self.n-1 != l_index:
                start = time.time()
                dec = Decoder()
                fail_index = l_index
                fail_slice = dec(sorted_results)
                end = time.time()
                decode_time = end-start
                del results_dict[self.n-1]
                results_dict[fail_index] = fail_slice
                sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
        elif self.type == "naive":
            # Sort and merge results based on worker index
            l_index = max(results_dict.keys())
            del results_dict[l_index]
            
            sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]

        comp_time = max(comp_times.values())
        
        C = np.vstack(sorted_results)

        print("Resulting matrix C:")
        print(C)
        print("total time: {} ms".format((comp_time + decode_time)*1000))
        charm.exit()

def main(args):
    # Read the configuration file
    conf = Config("./conf.json")
    
    # Load matrix dimensions
    A_rows = conf.cfg['A_row']
    A_cols_B_rows = conf.cfg['A_col_B_row']
    B_cols = conf.cfg['B_col']
    type = conf.cfg['type']
    fail_index = conf.cfg['fail_index']
    struggle_time = conf.cfg['struggle_time']

    # Number of workers (processors)
    n = conf.cfg['processor_num']
    
    # Create the matrix generator by np.random.rand()
    m_generator = MatrixGenerator(np.random.rand)

    # Generate random matrices A and B
    A = m_generator(A_rows, A_cols_B_rows)
    B = m_generator(A_cols_B_rows, B_cols)

    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Correct Matrix C:")
    print(np.dot(A, B))

    # Start the main process
    main_proxy = Chare(Main)
    main_proxy.start(n, A, B, type, fail_index, struggle_time)

charm.start(main)

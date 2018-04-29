from joblib import Parallel, delayed
import multiprocessing
import numpy as np
def calc(i):
    return (i, [i, 2*i, 3*i, 4*i])

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores)(delayed(calc)(class_name) for class_name in range(80))
    print(res)
    species = [x for x, _ in res]
    predictions = np.array([y for _, y in res])
    print(species)
    print(predictions)
    assert len(species) == 3
    assert len(predictions[0]) == 4
    result = predictions.T
    print(result)
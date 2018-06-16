from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import threading

def top_k_acc(y_predicted, y_true, class_map, k):
    '''Calculates the top_k-accuracy for the prediction of xgboost.'''

    count_matching_species = 0
    for i in range(len(y_predicted)):
        pred = y_predicted[i]
        _, sorted_species = zip(*reversed(sorted(zip(pred, list(class_map)))))
        if y_true[i] in sorted_species[:k]:
            count_matching_species += 1

    return count_matching_species / len(y_predicted)

class top_k_accuracy():
    '''Class was a try to speed up calculation with multicore but takes more time in the end.'''
    def get_score(self, y_predicted, y_true, class_map, k):
        self.y_true = y_true
        self.class_map = class_map
        self.k = k
        self.y_predicted = y_predicted

        jobs = []

        for i in tqdm(range(len(self.y_predicted))):
            count_matching_species = 0
            #process = mp.Process(target=self.get_result, args=(i, count_matching_species))
            #jobs.append(process)
            thread = threading.Thread(target=self.get_result, args=(i, count_matching_species))
            jobs.append(thread)

        # Start the processes (i.e. calculate the random number lists)
        for j in tqdm(jobs):
            j.start()

        # Ensure all of the processes have finished
        for j in tqdm(jobs):
            j.join()

        return count_matching_species / len(self.y_predicted)

    def get_result(self, i, count_matching_species):
        pred = self.y_predicted[i]
        _, sorted_species = zip(*reversed(sorted(zip(pred, list(self.class_map)))))
        if self.y_true[i] in sorted_species[:self.k]:
            count_matching_species += 1
import KerasModel
import submission_maker
import evaluation
import DataReader
import settings
import time

def full_run():
    start_time = time.time()

    DataReader.read_and_write_data()
    KerasModel.run_Model()
    submission_maker.make_submission()
    evaluation.evaluate_with_mrr()

    print("Total duration:", time.time() - start_time, "s")

if __name__ == '__main__':
    full_run()
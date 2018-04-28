#contains metrics functions for the keras model

from keras.metrics import top_k_categorical_accuracy
import functools


def get_top3_accuracy():
    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    return top3_acc

def get_top10_accuracy():
    top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
    top10_acc.__name__ = 'top10_acc'
    return top10_acc

def get_top50_accuracy():
    top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
    top50_acc.__name__ = 'top50_acc'
    return top50_acc
from math import sqrt

def get_vector_length(v):
    summ = 0

    for num in v:
        summ += num * num

    distance = sqrt(summ)
    
    return distance

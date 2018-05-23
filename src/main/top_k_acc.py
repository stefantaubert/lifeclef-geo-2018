from tqdm import tqdm

def top_k_acc(y_predicted, y_true, class_map, k):
    count_matching_species = 0
    for i in tqdm(range(len(y_predicted))):
        pred = y_predicted[i]
        _, sorted_species = zip(*reversed(sorted(zip(pred, list(class_map)))))
        if y_true[i] in sorted_species[:k]:
            count_matching_species += 1

    return count_matching_species / len(y_predicted)
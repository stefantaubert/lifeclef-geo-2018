from itertools import chain
from tqdm import tqdm

def get_ranks_df(submission_df, solutions, top_n):
    '''
    Returns all the ranks of the correct predicted species.
    If the right species is not in the first top_n-predictions, rank zero is added.

    Keyword arguments:
    submission_df -- contains the submission DataFrame which was created with submission_maker (consists of columns: glc_id, species, prediction, ranks)
    solutions -- contains an array with the right species_id for each glc_id in submissions
    top_n -- a number which specifies how many of the first submissions for each glc_id should be looked at to get the rank or if not contained rank zero.
    '''

    submission_matrix = submission_df.as_matrix()
    return get_ranks(submission_matrix, solutions, top_n)

def get_ranks(submissions, solutions, top_n):
    '''
    Returns all the ranks of the correct predicted species.
    If the right species is not in the first top_n-predictions, rank zero is added.

    Keyword arguments:
    submissions -- contains the submission array which was created with submission_maker (consists of columns: glc_id, species, prediction, ranks)
    solutions -- contains an array with the right species_id for each glc_id in submissions
    top_n -- a number which specifies how many of the first submissions for each glc_id should be looked at to get the rank or if not contained rank zero.
    '''
    
    assert len(submissions) % top_n == 0
    assert len(submissions) / top_n == len(solutions)

    # creates an array which contains the species for each submissionrow, for example [3,3,3,4,4,4] for 3 classes and 2 predictions (and top_n = 2)
    sol_array = [[s] * top_n for s in solutions]
    sol_array = list(chain.from_iterable(sol_array))
    assert len(sol_array) == len(submissions)

    count_patch_ids = len(solutions)
    ranks = []

    for i in tqdm(range(count_patch_ids)):
        current_start_index = i * top_n
        rank_for_patch_id = 0
        for j in range(top_n):
            current_index = current_start_index + j
            current_row = submissions[current_index]
            current_species = int(current_row[1])
            current_solution = sol_array[current_index]

            if current_species == current_solution:            
                current_rank = current_row[3]
                rank_for_patch_id = int(current_rank)
                break
                
        ranks.append(rank_for_patch_id)

    return ranks

from tqdm import tqdm

def map_groups(original, groups):
        name_group_dict = {}
        for name, group in groups.items():
            for species in group:
                name_group_dict[species] = name
        
        result = list(original)
        for i in tqdm(range(len(result))):
            current_species = result[i]
            ### all species have to be in one group
            assert current_species in name_group_dict.keys()
            result[i] = name_group_dict[current_species]
        
        return result

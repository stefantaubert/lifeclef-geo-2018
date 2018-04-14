from tqdm import tqdm
class SpeciesGroup():
    ### alt

    def get_groups(self, dictionary):
        for key, val in dictionary.items():
            assert len(val) == len(set(val))
            assert key not in val
        self.processed_species = []
        result = []
        processed = []
        for species_id, neighbor_species in tqdm(dictionary.items()):
            if species_id not in processed:
                children = self.get_species_group(species_id, dictionary)
                processed.extend(children)
                result.append(children)
                
        return result
    
    processed_species = []

    def iter(self, dictionary):
        for key, val in dictionary.items():
            assert len(val) == len(set(val))
            assert key not in val

        groups = []
        processed_species = []
        
        for current_species_id, current_neighbor_species in tqdm(dictionary.items()):
            if current_species_id not in processed_species:
                current_group = set(current_neighbor_species)
                for other_species_id, other_neighbor_species in dictionary.items():
                    if current_species_id != other_species_id:
                        tmp = set(other_neighbor_species)
                        has_atleast_one_similar_species = len(current_group.intersection(tmp)) > 0
                        if has_atleast_one_similar_species:
                            current_group = current_group.union(tmp)
                groups.append(current_group)
                processed_species.extend(current_group)
        
        final_groups = []
        for current_group in groups:
            new_group = current_group
            for other_group in groups: 
                if new_group != other_group:
                    has_atleast_one_similar_species = len(current_group.intersection(other_group)) > 0
                    new_group = new_group.union(other_group)
            final_groups.append(new_group)

        return groups

    def get_species_group(self, species, dic):
        assert species in dic.keys()
        if species in self.processed_species:
            return set([])
        else:
            self.processed_species.append(species)
            result = set([])
            current_neighbors = set(dic[species] + [species])
            result = current_neighbors
            for neighbor_species in current_neighbors:
                result = result.union(self.get_species_group(neighbor_species, dic))
            return result

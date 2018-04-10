from tqdm import tqdm

class SpeciesGroup():

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

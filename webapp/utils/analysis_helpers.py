import config

def get_triplet_by_id(triplet_id):
    for triplet in config.triplets_data:
        if triplet.get('id') == triplet_id:
            return triplet
    return None 
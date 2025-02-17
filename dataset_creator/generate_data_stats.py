import pandas as pd
import numpy as np



def get_dataset_statistics(csv_file_path: str):
    features = [
        "danceability",
        "mood_acoustic",
        "mood_aggressive",
        "mood_electronic",
        "mood_happy",
        "mood_party",
        "mood_relaxed",
        "mood_sad"
    ]

    df = pd.read_csv(csv_file_path).round(20).drop_duplicates()
    mood_counts = df[features].idxmax(axis=1).value_counts().to_dict()





    return mood_counts

def get_dataset_statistics_from_comprehensive_mood(csv_file_path: str = "/Volumes/Drive/MoodySound/data/metadata.csv"):
    metadata_df = pd.read_csv(csv_file_path)
    print(str(len(metadata_df)) + ' samples')


    features = {
        "danceability": 0,
        "mood_acoustic": 0,
        "mood_aggressive": 0,
        "mood_electronic": 0,
        "mood_happy": 0,
        "mood_party": 0,
        "mood_relaxed": 0,
        "mood_sad": 0
    }

    for feature in features.keys(): 
        features[feature] = metadata_df[metadata_df.comprehensive_mood == feature].shape[0]

    
    print(features)

            
if __name__ == "__main__":

    #stats = get_dataset_statistics("dataset_creator/augmented.csv")
    #print(stats)
    #stats = get_dataset_statistics_from_comprehensive_mood("/Volumes/Drive/MoodySound/data/metadata.csv")
    stats = get_dataset_statistics_from_comprehensive_mood("/Users/rishi/MoodySound2/dataset_creator/shuffled_metadata.csv")




    """
    Testing

    # danceability,mood_acoustic,mood_aggressive,mood_electronic,mood_happy,mood_party,mood_relaxed,mood_sad
    mood =  {"danceability": 0.0758428892226954,
            "mood_acoustic": 0.04758139805407882,
            "mood_aggressive": 0.04940619751761436,
            "mood_electronic": 0.3242922245377811,
            "mood_happy": 0.06552599845630962,
            "mood_party": 0.31666146070749207,
            "mood_relaxed": 0.021939055662141476,
            "mood_sad": 0.09875077584188718}

    stats = get_dataset_statistics("/Users/rishi/MoodySound/dataset_creator/base_moods.csv")
    print(stats)

    stats = get_dataset_statistics("/Users/rishi/MoodySound/dataset_creator/big_data_mood.csv")
    print(stats)

    stats = get_dataset_statistics("/Users/rishi/MoodySound2/dataset_creator/augmented.csv")
    print(stats)

    """


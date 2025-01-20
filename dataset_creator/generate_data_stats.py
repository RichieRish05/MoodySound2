import pandas as pd



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

    df = pd.read_csv(csv_file_path).round(10)
    mood_counts = df[features].idxmax(axis=1).value_counts().to_dict()





    return mood_counts




stats = get_dataset_statistics("/Users/rishi/MoodySound/dataset_creator/augmented.csv")
print(stats)




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

"""

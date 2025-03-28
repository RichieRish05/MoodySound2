import pandas as pd

def create_new_ratios(csv_path, ratios):
    features = [
        "mood_acoustic",
        "mood_aggressive",
        "mood_electronic",
        "mood_happy",
        "mood_party",
        "mood_relaxed",
        "mood_sad"
    ]

    df = pd.read_csv(csv_path).drop_duplicates()
    total_rows = len(df)

    # Calculate target counts for each mood based on ratios
    ratio_count = {mood: int(ratio * total_rows) for mood, ratio in ratios.items()}

    print(ratio_count)
    # Get current counts for each mood
    mood_counts = df["dominant_mood"].value_counts().to_dict()

    print(mood_counts)
    
    # Initialize result DataFrame
    result_df = pd.DataFrame()
    
    # For each mood, sample the appropriate number of rows
    for mood in features:
        current_count = mood_counts[mood]
        target_count = ratio_count[mood]
        
        # Get rows where this mood is dominant
        mood_rows = df[df["dominant_mood"] == mood]
        
        if current_count > target_count:
            # If we have too many, randomly sample down
            mood_rows = mood_rows.sample(n=target_count, random_state=42)
        elif current_count < target_count:
            # If we have too few, sample with replacement
            mood_rows = mood_rows.sample(n=target_count, replace=True, random_state=42)
            
        result_df = pd.concat([result_df, mood_rows])
    
    # Shuffle the final dataset
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return result_df


def get_statistics(csv_path):
    df = pd.read_csv(csv_path)

    mood_counts = df["dominant_mood"].value_counts().to_dict()

    print(mood_counts)

if __name__ == '__main__':
    ratios = {
        "mood_danceable": 0.04,
        "mood_acoustic": 0.16,
        "mood_aggressive": 0.22,
        "mood_electronic": 0.04,
        "mood_happy": 0.14,
        "mood_party": 0.10,
        "mood_relaxed": 0.18,
        "mood_sad": 0.12
    }

    csv_path = '/Users/rishi/MoodySound2/csv_modifier/metadata.csv'


    # new_df = create_new_ratios(csv_path, ratios)
    # new_df.to_csv('new.csv')

    get_statistics('/Users/rishi/MoodySound2/csv_modifier/new.csv')

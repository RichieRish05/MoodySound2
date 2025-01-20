from pathlib import Path
import json 
import csv

        

def extract_features(file_path: Path):
    """
    Extract all features from a single json file
    """
    # The metadata
    metadata = ["title",
                "artist"]

    # All the features of our mood vector
    mood_features = ["danceability",
                    "mood_acoustic",
                    "mood_aggressive", 
                    "mood_electronic", 
                    "mood_happy", 
                    "mood_party", 
                    "mood_relaxed",
                    "mood_sad"]
    
    # Initialize a dictionary to represent a row in our csv

    csv_row = {}
    mood_vector = {}
    try:
        with open(file_path) as f:
            data = json.load(f)

            for info in metadata:
                csv_row[info] = data["metadata"]["tags"][info][0]

            for feature in mood_features:
                if feature in data["highlevel"]:
                    point = next(iter(data["highlevel"][feature]["all"].values()))
                    mood_vector[feature] = point
            
                
            csv_row = csv_row | mood_vector
            # If the csv row contains all necessary features, it is returned
            if len(csv_row.keys()) == len(metadata) + len(mood_features):
                print(f'Succesfully processed {file_path}')
                return(csv_row)
            else:
                print('Error processing {file_path}: could not extract all features')
            
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
        print(f'Error processing {file_path}: {e}')
        return None


def recursive_extract(directory_path: Path, writer: csv.DictWriter):
    """
    Recursively write all features to a csv file from all json files
    within a directory
    """
    for file in directory_path.iterdir():
        if file.is_dir():
            recursive_extract(file, writer)
        elif not file.is_file() or file.suffix != ".json":
            continue
        else:
            data = extract_features(file)
            if data: 
                writer.writerow(data)


def create_csv_file(directory_path: Path):
    """
    Creates a csv file, extract all features from all json files within a directory recursively.
    and write it all to the csv file
    """

    headers = ["title",
               "artist",
               "danceability",
               "mood_acoustic",
               "mood_aggressive", 
               "mood_electronic", 
               "mood_happy",
               "mood_party", 
               "mood_relaxed",
               "mood_sad"]
    
    with open("CHANGE HERE", mode='w', newline='') as file:
        # Create a CSV DictWriter object
        writer = csv.DictWriter(file, fieldnames=headers)

        # Write the header 
        writer.writeheader()
    
        # Recursively extract features from all json files within directory
        recursive_extract(directory_path, writer)
                    
    

# Work with this part of the data first
directory_path = Path("/Users/rishi/MoodySound/dataset_creator/acousticbrainz-highlevel-json-20220623/highlevel")
create_csv_file(directory_path)


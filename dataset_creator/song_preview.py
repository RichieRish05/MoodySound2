import requests

#Deezer API base url
BASE_URL = "https://api.deezer.com"


def get_song_preview(query: str) -> str:
    # Attempt to contact the API endpoint
    try:
        # Encode the url request
        params = {'q': query}
        search_url = f'{BASE_URL}/search'

        # Make the request
        response = requests.get(search_url, params=params)

        # Raise an exception if the response status is not 200
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Find and return the preview
        track = data["data"][0]
        if track["preview"]:
            return track["preview"]
        
        # In case there is no preview
        return None
    
    # Catch any errors
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    

# Testing
# print(get_song_preview("Hello"))

    
    


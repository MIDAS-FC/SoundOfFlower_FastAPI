import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List

# .env 파일의 환경 변수를 불러옴.
load_dotenv()

class Song(BaseModel):
    trackId : str
    trackName : str
    artistName : str

def get_songs(playlistID:str) -> List[Song]:
    cid = os.getenv('SPOTIFY_CID')
    secret = os.getenv('SPOTIFY_SECRET')
    # 기존에는 37i9dQZF1DXcBWIGoYBM5M
    playlist_id = playlistID
    songs = []
    
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)    
    results = sp.playlist_items(playlist_id, market='KR')

    for item in results['items']:
        track = item['track']
        artist_names = [artist['name'] for artist in track['artists']]
        data = {
            "trackId": str(track['id']),
            "trackName": str(track['name']),
            "artistName": str(', '.join(artist_names))
        }
        song = Song(**data)
        songs.append(song)
        
    return songs


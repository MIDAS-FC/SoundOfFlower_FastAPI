import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
import base64
import requests

load_dotenv()

class Song(BaseModel):
    trackId : str
    trackName : str
    artistName : str

def get_songs(playlistID:str) -> List[Song]:
    cid = os.getenv('SPOTIFY_CID')
    secret = os.getenv('SPOTIFY_SECRET')
    playlist_id = playlistID
    songs = []
    
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)    
    results = sp.playlist_items(playlist_id, market='KR')

    for item in results['items']:
        track = item['track']
        is_playble = track['is_playable']
        explicit = track['explicit']
        print("is playble? "+str(is_playble))
        print("explicit? "+str(explicit))
        if is_playble == False:
            continue
        if explicit == True:
            continue
        if track['preview_url'] is None:
            print("preview url이 없어용")
            continue
        
        print("after if")
        artist_names = [artist['name'] for artist in track['artists']]
        data = {
            "trackId": str(track['id']),
            "trackName": str(track['name']),
            "artistName": str(', '.join(artist_names))
        }
        song = Song(**data)
        songs.append(song)
        
    return songs

def get_spotify_token():
    SPOTIFY_CID = os.getenv('SPOTIFY_CID')
    SPOTIFY_SECRET = os.getenv('SPOTIFY_SECRET')

    auth_url = 'https://accounts.spotify.com/api/token'

    auth_header = base64.b64encode(f"{SPOTIFY_CID}:{SPOTIFY_SECRET}".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    
    response = requests.post(auth_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        return None
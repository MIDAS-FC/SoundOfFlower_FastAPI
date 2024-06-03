import requests
import json
from dotenv import load_dotenv
import os
from spotify.spotifyAPI import Song
from typing import List
import nltk
from nltk.corpus import stopwords
import re

load_dotenv()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_lyrics(song: Song) -> str:
    
    api_key = "&apikey=" + os.getenv('MUSIXMATCH_API_KEY')

    base_url = "https://api.musixmatch.com/ws/1.1/"
    lyrics_matcher = "matcher.lyrics.get"
    format_url = "?format=json&callback=callback"
    artist_search_parameter = "&q_artist="
    track_search_parameter = "&q_track="

    artist_name = song.artistName
    track_name = song.trackName
    print("="*50)
    print("artist_name : "+artist_name)
    print("track_name : "+track_name)

    api_call = (base_url + lyrics_matcher + format_url + artist_search_parameter + artist_name
                + track_search_parameter + track_name + api_key)

    request = requests.get(api_call)
    data = request.json()
    status_code = data['message']['header']['status_code']
    
    if status_code == 200:
        data = data['message']['body']
        strs = data['lyrics']['lyrics_body'].split('\n')
        totalStr = ""
        for str in strs:
            simple_preprocess(str)
            totalStr = totalStr+str+". " 
        return totalStr
    else:
        return None

def simple_preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) #특수문자 및 숫자 제거
    text = text.lower() #소문자로 변환
    text = ' '.join([word for word in text.split() if word not in stop_words]) #stopwords 제거
    return text
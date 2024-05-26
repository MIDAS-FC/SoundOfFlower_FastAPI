import requests
import json
from dotenv import load_dotenv
import os

# .env 파일의 환경 변수를 불러옴.
load_dotenv()

api_key = "&apikey=" + os.getenv('MUSIXMATCH_API_KEY')

base_url = "https://api.musixmatch.com/ws/1.1/"
lyrics_matcher = "matcher.lyrics.get"
format_url = "?format=json&callback=callback"
artist_search_parameter = "&q_artist="
track_search_parameter = "&q_track="

artist_name = "aespa"
track_name = "supernova"

api_call = (base_url + lyrics_matcher + format_url + artist_search_parameter + artist_name
            + track_search_parameter + track_name + api_key)

# call the api
request = requests.get(api_call)
data = request.json()
data = data['message']['body']
print(data['lyrics']['lyrics_body'])

strs = data['lyrics']['lyrics_body'].split('\n')

# split 했을 때 마지막 3개 데이터 필요없음.
for(index,string) in enumerate(strs):
    print(index)
    print(string)

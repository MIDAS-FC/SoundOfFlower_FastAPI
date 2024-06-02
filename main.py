import torch
import numpy as np
import emotionFunc
from DB.database import engineconn
from fastapi import FastAPI, Request
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from kr.positive_bert import PositiveBERTClassifier
from kr.negative_bert import NegativeBERTClassifier
from kr.bertDataSet import BERTDataset
from starlette.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from kobert_tokenizer import KoBERTTokenizer
from spotify.spotifyAPI import get_songs, get_spotify_token
from musixmatch.musixmatchAPI import get_lyrics
from pydanticModels import SongItem
import createSong
import torch.nn.functional as F


app = FastAPI()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # 모델 로드
# positive_model_dict = torch.load('positive.pt', map_location=device)
# negative_model_dict = torch.load('negative.pt', map_location=device)

# # 긍정/부정에 공통적으로 쓰이는 BERT 모델과 토크나이저 설정
# bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
# kr_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# # 긍정 모델 초기화 및 로드
# positive_model = PositiveBERTClassifier(bertmodel, dr_rate=0.5).to(device)
# positive_model.load_state_dict(positive_model_dict)

# # 부정 모델 초기화 및 로드
# negative_model = NegativeBERTClassifier(bertmodel, dr_rate=0.5).to(device)
# negative_model.load_state_dict(negative_model_dict)

#긍정/부정에 공통적으로 쓰이는
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
kr_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

#긍정
positive_model_dict = torch.load('positive.pt')
positive_model = PositiveBERTClassifier(bertmodel,  dr_rate = 0.5).to(device)
positive_model.load_state_dict(positive_model_dict)

#부정
negative_model_dict = torch.load('negative.pt')
negative_model = NegativeBERTClassifier(bertmodel,  dr_rate = 0.5).to(device)
negative_model.load_state_dict(negative_model_dict)

origins = [ "*" ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware

engine = engineconn()
session = engine.sessionmaker()

@app.post("/analyze/emotion")
async def predict(input:Request):
    input = await input.json()
    inputStr = input['comment']
    inputEmotion = input['emotion']
    inputMaintain = input['maintain']
    
    #주어진 문장과 더미 레이블을 포함하는 데이터를 준비
    data = [inputStr, '0']  # '0'은 더미 레이블
    dataset_another = [data]
    #주어진 데이터로 BERTDataset 초기화
    another_test = BERTDataset(dataset_another, 0, 1, kr_tokenizer, 64, True, False)
    #DataLoader를 생성하여 데이터셋을 배치로 나누고 모델에 제공
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)
    
    #emotion이 뭔지? -> 각 이모션에 맞는 모델로 감정 분석, 
    if inputEmotion == 'positive':
        positive_emotion_counts = np.zeros(2)
        positive_emotions = ["기쁨", "사랑"]
        positive_model.eval()
        
            #배치 데이터를 반복. 각 배치에는 한 문장이 포함됨.
        for batch_data in test_dataloader:
            #배치 데이터로부터 토큰 ID와 어텐션 마스크 추출. 이것들은 모델에 입력됨.
            token_ids = batch_data['input_ids'].to(device) 
            attention_mask = batch_data['attention_mask'].to(device)

            #모델을 통해 문장을 전달해 예측 수행
            out = positive_model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))
            # Softmax를 적용하여 모델의 출력을 확률로 변환.
            logits = out.detach().cpu().numpy()
            #probabilities는 한 문장에 대해 각각의 감정 카테고리에 대한 확률을 포함하고 있음
            probabilities = softmax(torch.tensor(logits), dim=1).numpy() 
            positive_emotion_counts += probabilities[0]  # 각각의 감정에 대한 확률 값을 감정 카테고리별 카운트에 더한다.
            predicted_emotion = positive_emotions[np.argmax(probabilities)] #가장 높은 확률을 가진 감정을 츄출
            emotionList = [0, 0, positive_emotion_counts[0], 0, 0, 0, positive_emotion_counts[1]]
            # angry0, sad1, delight2, calm3, embarrased4(불안), anxiety5(우울), love6
    elif inputEmotion == 'negative':
        negative_emotion_counts = np.zeros(4)
        negative_emotions = ["분노", "우울", "불안", "슬픔"]
        negative_model.eval()
        
            #배치 데이터를 반복. 각 배치에는 한 문장이 포함됨.
        for batch_data in test_dataloader:
            #배치 데이터로부터 토큰 ID와 어텐션 마스크 추출. 이것들은 모델에 입력됨.
            token_ids = batch_data['input_ids'].to(device) 
            attention_mask = batch_data['attention_mask'].to(device)

            #모델을 통해 문장을 전달해 예측 수행
            out = negative_model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))
            # Softmax를 적용하여 모델의 출력을 확률로 변환.
            logits = out.detach().cpu().numpy()
            #probabilities는 한 문장에 대해 각각의 감정 카테고리에 대한 확률을 포함하고 있음
            probabilities = softmax(torch.tensor(logits), dim=1).numpy() 
            negative_emotion_counts += probabilities[0]  # 각각의 감정에 대한 확률 값을 감정 카테고리별 카운트에 더한다.
            predicted_emotion = negative_emotions[np.argmax(probabilities)] #가장 높은 확률을 가진 감정을 츄출
            emotionList = [negative_emotion_counts[0], negative_emotion_counts[3], 0, 0, negative_emotion_counts[2], negative_emotion_counts[1], 0]
            #[2]가 원래 당황, [1]이 불안
    else: #neutrality. 감정 분석 안함. 바로 중립 노래 추천    
        predicted_emotion = '중립'
        emotionList = [0, 0, 0, 1, 0, 0, 0]
    
    return {
        'flower' : emotionFunc.flower(predicted_emotion),
        'angry' : emotionList[0],
        'sad' : emotionList[1],
        'delight' : emotionList[2],
        'calm' : emotionList[3],
        'anxiety' : emotionList[4],
        'depressed' : emotionList[5],
        'love' : emotionList[6],
        'spotify' : emotionFunc.get_spotifyId(db=session, emotion=predicted_emotion, 
                                              sad=emotionList[1], delight=emotionList[2], love=emotionList[6], #sad, delight, love 정보만 필요 
                                              maintain=inputMaintain, preEmotion=inputEmotion)
    }
        
model = BertForSequenceClassification.from_pretrained("./eng_emotion_model")
tokenizer = BertTokenizer.from_pretrained("./eng_emotion_model")

@app.post("/soundOfFlower/updateDB")
async def updateDB(input:Request): 
    input = await input.json()
    inputPlaylist = input['playlist']
    if not inputPlaylist.strip():  # playlist가 없거나 공백으로 이루어져있으면
        return {"validInput": True}
    
    # 감정 레이블 정의
    emotion_labels = ['happy', 'love', 'sadness' ]
    emotion_type_labels = ['delight', 'love', 'sad']
    songs = get_songs(inputPlaylist)
    for song in songs:
        if createSong.alreadyExist(db=session, spotify=song.trackId): #이미 DB에 있으면 continue
            print("continue!")
            continue
        
        lyricList = get_lyrics(song)
        print("a")
        print(type(lyricList))
        if lyricList == None:
            continue
        
        total_emotion = np.zeros(len(emotion_labels))
        
        print("before lyric for")
        for lyric in lyricList:
            print("after lyric for")
            text = lyric
            processed_text = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**processed_text)
                scores = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
            total_emotion += scores
                
        average_emotion = total_emotion / len(lyricList)
        max_index = np.argmax(average_emotion)
        print("max_index : "+str(max_index))
        emotion = emotion_labels[max_index]
        
        songItem = SongItem(title=song.trackName, spotify=song.trackId, emotion=emotion, emotionList=average_emotion.tolist())
        print("emotion : "+emotion)
        createSong.create_Music(session, songItem, emotion_type_labels[max_index])
        if emotion == 'sadness':
            print("emotion : "+emotion)
            createSong.create_sadMusic(session, songItem)
        elif emotion == 'happy':
            print(emotion)
            createSong.create_delightMusic(session, songItem)
        elif emotion == 'love':
            print(emotion)
            createSong.create_loveMusic(session, songItem)
            
    return {"validInput": True}

@app.get("/soundOfFlower/getSpotifyToken")
async def getSpotifyToken():
    token = get_spotify_token()
    if token == None:
        return {"token" : None}
    else:
        return {"token" : token}


    

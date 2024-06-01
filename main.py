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
from spotify.spotifyAPI import get_songs
from musixmatch.musixmatchAPI import get_lyrics
from pydanticModels import SongItem
import createSong
import joblib

app = FastAPI()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

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
    
    emotion_counts = np.zeros(7)
    emotions = ["분노", "중립", "불안", "당황", "슬픔", "기쁨", "사랑"]
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
       
    elif inputEmotion == 'negative':
        negative_emotion_counts = np.zeros(4)
        negative_emotions = ["분노", "불안", "당황", "슬픔"]
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
        
    else: #neutrality. 감정 분석 안함. 바로 중립 노래 추천    
        predicted_emotion = '중립'
        emotionList = [0, 0, 0, 1, 0, 0, 0]
    #배치 데이터를 반복. 각 배치에는 한 문장이 포함됨.
    # for batch_data in test_dataloader:
    #     #배치 데이터로부터 토큰 ID와 어텐션 마스크 추출. 이것들은 모델에 입력됨.
    #     token_ids = batch_data['input_ids'].to(device) 
    #     attention_mask = batch_data['attention_mask'].to(device)

    #     #모델을 통해 문장을 전달해 예측 수행
    #     out = kr_model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))
    #     # Softmax를 적용하여 모델의 출력을 확률로 변환.
    #     logits = out.detach().cpu().numpy()
    #     #probabilities는 한 문장에 대해 각각의 감정 카테고리에 대한 확률을 포함하고 있음
    #     probabilities = softmax(torch.tensor(logits), dim=1).numpy() 
    #     emotion_counts += probabilities[0]  # 각각의 감정에 대한 확률 값을 감정 카테고리별 카운트에 더한다.
    #     predicted_emotion = emotions[np.argmax(probabilities)] #가장 높은 확률을 가진 감정을 츄출
    #     emotionList = emotion_counts.tolist()
    
    return {
        'flower' : emotionFunc.flower(predicted_emotion),
        'angry' : emotionList[0],
        'sad' : emotionList[1],
        'delight' : emotionList[2],
        'calm' : emotionList[3],
        'embarrased' : emotionList[4],
        'anxiety' : emotionList[5],
        'love' : emotionList[6],
        'spotify' : emotionFunc.get_spotifyId(db=session, emotion=predicted_emotion, 
                                              sad=emotionList[1], delight=emotionList[2], love=emotionList[6], #sad, delight, love 정보만 필요 
                                              maintain=inputMaintain, preEmotion=inputEmotion)
    }
        
model = joblib.load('emotion_classification_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  

# # 모델 구조 생성
# eng_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
# eng_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # 저장된 상태 사전을 모델에 불러오기
# eng_model_dict = torch.load('emotion_analysis_model_eng.pt')
# eng_model.load_state_dict(eng_model_dict['model_state_dict'])

@app.post("/soundOfFlower/updateDB")
async def updateDB(input:Request): 
    input = await input.json()
    inputPlaylist = input['playlist']
    if not inputPlaylist.strip():  # playlist가 없거나 공백으로 이루어져있으면
        return {"validInput": True}
    
    # 감정 레이블 정의
    emotion_labels = ['sadness', 'happiness', 'love']
    emotion_type_labels = ['sad', 'delight', 'love']
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
            text_vec = vectorizer.transform([text])
            scores = model.predict_proba(text_vec)[0]
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
        elif emotion == 'happiness':
            print(emotion)
            createSong.create_delightMusic(session, songItem)
        elif emotion == 'love':
            print(emotion)
            createSong.create_loveMusic(session, songItem)
            
    return {"validInput": True}



    

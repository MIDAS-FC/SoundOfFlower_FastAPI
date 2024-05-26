import torch
import numpy as np
import emotionFunc
from DB.database import engineconn
from fastapi import FastAPI, Request
from transformers import BertModel
from kr.bert import BERTClassifier
from kr.bertDataSet import BERTDataset
from starlette.middleware.cors import CORSMiddleware
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer

app = FastAPI()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
model_dict = torch.load('kobert_emotion_model.pt')
model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(device)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

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

@app.post("/soundOfFlower/predict")
async def predict(input:Request):
    input = await input.json()
    inputStr = input['inputStr']
    emotion_counts = np.zeros(7)
    emotions = ["분노", "중립", "불안", "당황", "슬픔", "기쁨", "사랑"]
    #주어진 문장과 더미 레이블을 포함하는 데이터를 준비
    data = [inputStr, '0']  # '0'은 더미 레이블
    dataset_another = [data]
    #주어진 데이터로 BERTDataset 초기화
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, 64, True, False)
    #DataLoader를 생성하여 데이터셋을 배치로 나누고 모델에 제공
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

    #평가 모드로 설정
    model.eval()
    #배치 데이터를 반복. 각 배치에는 한 문장이 포함됨.
    for batch_data in test_dataloader:
        #배치 데이터로부터 토큰 ID와 어텐션 마스크 추출. 이것들은 모델에 입력됨.
        token_ids = batch_data['input_ids'].to(device) 
        attention_mask = batch_data['attention_mask'].to(device)

        #모델을 통해 문장을 전달해 예측 수행
        out = model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))
        # Softmax를 적용하여 모델의 출력을 확률로 변환.
        logits = out.detach().cpu().numpy()
        #probabilities는 한 문장에 대해 각각의 감정 카테고리에 대한 확률을 포함하고 있음
        probabilities = softmax(torch.tensor(logits), dim=1).numpy() 
        emotion_counts += probabilities[0]  # 각각의 감정에 대한 확률 값을 감정 카테고리별 카운트에 더한다.
        predicted_emotion = emotions[np.argmax(probabilities)] #가장 높은 확률을 가진 감정을 츄출
        emotionList = emotion_counts.tolist()
        
        return {
            'flower' : emotionFunc.flower(predicted_emotion),
            'angry' : emotionList[0],
            'sad' : emotionList[4],
            'delight' : emotionList[5],
            'calm' : emotionList[1],
            'embarrased' : emotionList[3],
            'anxiety' : emotionList[2],
            'love' : emotionList[6],
            'musicId' : emotionFunc.get_spotifyId(session, predicted_emotion, emotionList[0], emotionList[4], emotionList[5], emotionList[1], emotionList[3], emotionList[2], emotionList[6])
        }

    

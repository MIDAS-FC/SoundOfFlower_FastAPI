from sqlalchemy.orm import Session
from getMusic import get_angryMusic, get_delightMusic, get_embrrasedMusic, get_anxietyMusic, get_calmMusic, get_loveMusic, get_sadMusic

def flower(emotion):
    if emotion == "분노":
        return "장미"
    elif emotion == "기쁨":
        return "해바라기"
    elif emotion == "당황":
        return "튤립"
    elif emotion == "불안":
        return "라일락"
    elif emotion == "슬픔":
        return "블루 데이지"
    elif emotion == "중립":
        return "캐모마일"
    elif emotion == "사랑":
        return "달리아"
    
def get_spotifyId(db: Session, emotion:str, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    if emotion == "분노":
        angryMusic = get_angryMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if angryMusic:
            return angryMusic.spotipyId
    elif emotion == "기쁨":
        delightMusic = get_delightMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if delightMusic:
            return delightMusic.spotipyId
    elif emotion == "당황":
        embrrasedMusic = get_embrrasedMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if embrrasedMusic:
            return embrrasedMusic.spotipyId
    elif emotion == "불안":
        anxietyMusic = get_anxietyMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if anxietyMusic:
            return anxietyMusic.spotipyId
    elif emotion == "슬픔":
        sadMusic = get_sadMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if sadMusic:
            return sadMusic.spotipyId
    elif emotion == "중립":
        calmMusic = get_calmMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if calmMusic:
            return calmMusic.spotipyId
    elif emotion == "사랑":
        loveMusic = get_loveMusic(db, angry, sad, delight, calm, embrrased, anxiety, love)
        if loveMusic:
            return loveMusic.spotipyId
    else:
        return "잘못된 감정입니다."
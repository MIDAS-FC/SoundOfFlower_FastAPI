from sqlalchemy.orm import Session
from getMusic import get_delightMusic, get_loveMusic, get_sadMusic, get_positiveMusic, get_randomMusic

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
    
def get_spotifyId(db: Session, emotion:str, sad:float, delight:float, love:float, maintain:str, preEmotion:str):
    if preEmotion == "positive":
        if emotion == "기쁨":
            delightMusic = get_delightMusic(db, sad, delight, love)
            if delightMusic:
                return delightMusic.spotipyId
            else:
                return None
        else: #사랑
            loveMusic = get_loveMusic(db, sad, delight, love)
            if loveMusic:
                return loveMusic.spotipyId
            else:
                return None
    elif preEmotion == "neutrality":
        neutralMusic = get_randomMusic(db)
        if neutralMusic:
            return neutralMusic.spotipyId
        else:
            return None
    else: #negative
        if emotion == "슬픔":
            if maintain == "true":
                sadMusic = get_sadMusic(db, sad, delight, love)
                if sadMusic:
                    return sadMusic.spotipyId
                else:
                    return None
            else: # 긍정 노래 추천
                positiveMusic = get_positiveMusic(db)
                if positiveMusic:
                    return positiveMusic.spotipyId
                else:
                    return None    
        else: # 분노, 불안, 당황
            positiveMusic = get_positiveMusic(db)
            if positiveMusic:
                return positiveMusic.spotipyId
            else:
                return None
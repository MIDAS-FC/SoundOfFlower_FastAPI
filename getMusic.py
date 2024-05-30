from sqlalchemy.orm import Session
from sqlalchemy import func
from DB.models import SadMusic, DelightMusic, LoveMusic
import random

#슬픔 노래 추천
def get_sadMusic(db: Session, angry:float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(SadMusic).\
        order_by(
            func.sqrt(
                func.pow(SadMusic.angry - angry, 2) +
                func.pow(SadMusic.sad - sad, 2) +
                func.pow(SadMusic.delight - delight, 2) +
                func.pow(SadMusic.calm - calm, 2) +
                func.pow(SadMusic.embrrased - embrrased, 2) +
                func.pow(SadMusic.anxiety - anxiety, 2) +
                func.pow(SadMusic.love - love, 2)
            )
        ).\
        first()
    return closest_music

#사랑 및 기쁨(긍정) 노래 추천 -> 랜덤
def get_positiveMusic(db: Session):
    random_integer = random.randint(0, 1)
    if random_integer == 0: #사랑 노래 추천
        return db.query(LoveMusic).order_by(func.rand()).first()
    else: # 기쁨 노래 추천
        return db.query(DelightMusic).order_by(func.rand()).first()
    
#사랑 및 기쁨, 슬픔(랜덤) 노래 추천 -> 랜덤
def get_randomMusic(db: Session):
    random_integer = random.randint(0,2)
    if random_integer == 0: #사랑 노래 추천
        return db.query(LoveMusic).order_by(func.rand()).first()
    elif random_integer == 1: # 기쁨 노래 추천
        return db.query(DelightMusic).order_by(func.rand()).first()
    else : # 슬픔 노래 추천
        return db.query(SadMusic).order_by(func.rand()).first()
    
#기쁨 노래 추천
def get_delightMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(DelightMusic).\
        order_by(
            func.sqrt(
                func.pow(DelightMusic.angry - angry, 2) +
                func.pow(DelightMusic.sad - sad, 2) +
                func.pow(DelightMusic.delight - delight, 2) +
                func.pow(DelightMusic.calm - calm, 2) +
                func.pow(DelightMusic.embrrased - embrrased, 2) +
                func.pow(DelightMusic.anxiety - anxiety, 2) +
                func.pow(DelightMusic.love - love, 2)
            )
        ).\
        first()
    return closest_music

#사랑 노래 추천
def get_loveMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(LoveMusic).\
        order_by(
            func.sqrt(
                func.pow(LoveMusic.angry - angry, 2) +
                func.pow(LoveMusic.sad - sad, 2) +
                func.pow(LoveMusic.delight - delight, 2) +
                func.pow(LoveMusic.calm - calm, 2) +
                func.pow(LoveMusic.embrrased - embrrased, 2) +
                func.pow(LoveMusic.anxiety - anxiety, 2) +
                func.pow(LoveMusic.love - love, 2)
            )
        ).\
        first()
    return closest_music


# def create_user(db: Session, user: schemas.UserCreate):
#     db_user = models.User(name=user.name, email=user.email)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user
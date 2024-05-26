from sqlalchemy.orm import Session
from sqlalchemy import func
from DB.models import SadMusic, AngryMusic, AnxietyMusic, DelightMusic, CalmMusic, EmbrrasedMusic, LoveMusic

def get_sadMusic(db: Session, angry:float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(SadMusic).\
        order_by(
            func.abs(SadMusic.angry - angry) +
            func.abs(SadMusic.sad - sad) +
            func.abs(SadMusic.delight - delight) +
            func.abs(SadMusic.calm - calm) +
            func.abs(SadMusic.embrrased - embrrased) +
            func.abs(SadMusic.anxiety - anxiety) +
            func.abs(SadMusic.love - love)
        ).\
        first()
    return closest_music

def get_angryMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(AngryMusic).\
        order_by(
            func.abs(AngryMusic.angry - angry) +
            func.abs(AngryMusic.sad - sad) +
            func.abs(AngryMusic.delight - delight) +
            func.abs(AngryMusic.calm - calm) +
            func.abs(AngryMusic.embrrased - embrrased) +
            func.abs(AngryMusic.anxiety - anxiety) +
            func.abs(AngryMusic.love - love)
        ).\
        first()
    return closest_music

def get_delightMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(DelightMusic).\
        order_by(
            func.abs(DelightMusic.angry - angry) +
            func.abs(DelightMusic.sad - sad) +
            func.abs(DelightMusic.delight - delight) +
            func.abs(DelightMusic.calm - calm) +
            func.abs(DelightMusic.embrrased - embrrased) +
            func.abs(DelightMusic.anxiety - anxiety) +
            func.abs(DelightMusic.love - love)
        ).\
        first()
    return closest_music

def get_calmMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(CalmMusic).\
        order_by(
            func.abs(CalmMusic.angry - angry) +
            func.abs(CalmMusic.sad - sad) +
            func.abs(CalmMusic.delight - delight) +
            func.abs(CalmMusic.calm - calm) +
            func.abs(CalmMusic.embrrased - embrrased) +
            func.abs(CalmMusic.anxiety - anxiety) +
            func.abs(CalmMusic.love - love)
        ).\
        first()
    return closest_music

def get_embrrasedMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(EmbrrasedMusic).\
        order_by(
            func.abs(EmbrrasedMusic.angry - angry) +
            func.abs(EmbrrasedMusic.sad - sad) +
            func.abs(EmbrrasedMusic.delight - delight) +
            func.abs(EmbrrasedMusic.calm - calm) +
            func.abs(EmbrrasedMusic.embrrased - embrrased) +
            func.abs(EmbrrasedMusic.anxiety - anxiety) +
            func.abs(EmbrrasedMusic.love - love)
        ).\
        first()
    return closest_music

def get_anxietyMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(AnxietyMusic).\
        order_by(
            func.abs(AnxietyMusic.angry - angry) +
            func.abs(AnxietyMusic.sad - sad) +
            func.abs(AnxietyMusic.delight - delight) +
            func.abs(AnxietyMusic.calm - calm) +
            func.abs(AnxietyMusic.embrrased - embrrased) +
            func.abs(AnxietyMusic.anxiety - anxiety) +
            func.abs(AnxietyMusic.love - love)
        ).\
        first()
    return closest_music

def get_loveMusic(db: Session, angry: float, sad:float, delight:float, calm:float, embrrased:float, anxiety:float, love:float):
    closest_music = db.query(LoveMusic).\
        order_by(
            func.abs(LoveMusic.angry - angry) +
            func.abs(LoveMusic.sad - sad) +
            func.abs(LoveMusic.delight - delight) +
            func.abs(LoveMusic.calm - calm) +
            func.abs(LoveMusic.embrrased - embrrased) +
            func.abs(LoveMusic.anxiety - anxiety) +
            func.abs(LoveMusic.love - love)
        ).\
        first()
    return closest_music


# def create_user(db: Session, user: schemas.UserCreate):
#     db_user = models.User(name=user.name, email=user.email)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user
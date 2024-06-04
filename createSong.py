from sqlalchemy.orm import Session
from DB.models import SadMusic, DelightMusic, LoveMusic, Music
from pydanticModels import SongItem

def alreadyExist(db: Session, spotify:str):
    song = db.query(Music).filter(Music.spotify == spotify).first()
    if song:
        return True
    
    # song = db.query(SadMusic).filter(SadMusic.spotify == spotify).first()
    # if song:
    #     return True
    # song = db.query(DelightMusic).filter(DelightMusic.spotify == spotify).first()
    # if song:
    #     return True
    # song = db.query(LoveMusic).filter(LoveMusic.spotify == spotify).first()
    # if song:
    #     return True
    
    return False

def create_Music(db: Session, songItem:SongItem, emotionType:str):
    music = Music(spotify=songItem.spotify,
                  emotion_type=emotionType,
                  total_likes = 0)
    
    db.add(music)
    db.commit()
    db.refresh(music)

def create_sadMusic(db: Session, songItem:SongItem):
    sadmusic = SadMusic(spotify=songItem.spotify,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(sadmusic)
    db.commit()
    db.refresh(sadmusic)
    
def create_delightMusic(db: Session, songItem:SongItem):
    delightMusic = DelightMusic(spotify=songItem.spotify,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(delightMusic)
    db.commit()
    db.refresh(delightMusic)
    
def create_loveMusic(db: Session, songItem:SongItem):
    loveMusic = LoveMusic(spotify=songItem.spotify,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(loveMusic)
    db.commit()
    db.refresh(loveMusic)

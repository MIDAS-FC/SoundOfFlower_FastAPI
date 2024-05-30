from sqlalchemy.orm import Session
from DB.models import SadMusic, DelightMusic, LoveMusic
from pydanticModels import SongItem

def alreadyExist(db: Session, spotifyId:str):
    song = db.query(SadMusic).filter(SadMusic.spotipyId == spotifyId).first()
    if song:
        return True
    song = db.query(DelightMusic).filter(DelightMusic.spotipyId == spotifyId).first()
    if song:
        return True
    song = db.query(LoveMusic).filter(LoveMusic.spotipyId == spotifyId).first()
    if song:
        return True
    
    return False


def create_sadMusic(db: Session, songItem:SongItem):
    sadmusic = SadMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(sadmusic)
    db.commit()
    db.refresh(sadmusic)
    
def create_delightMusic(db: Session, songItem:SongItem):
    delightMusic = DelightMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(delightMusic)
    db.commit()
    db.refresh(delightMusic)
    
def create_loveMusic(db: Session, songItem:SongItem):
    loveMusic = LoveMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        sad = songItem.emotionList[0],
                        delight = songItem.emotionList[1],
                        love = songItem.emotionList[2])
    
    db.add(loveMusic)
    db.commit()
    db.refresh(loveMusic)

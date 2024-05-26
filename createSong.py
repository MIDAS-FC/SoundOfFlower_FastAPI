from sqlalchemy.orm import Session
from DB.models import SadMusic, AngryMusic, AnxietyMusic, DelightMusic, CalmMusic, EmbrrasedMusic, LoveMusic
from pydanticModels import SongItem

def create_sadMusic(db: Session, songItem:SongItem):
    sadmusic = SadMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(sadmusic)
    db.commit()
    db.refresh(sadmusic)
    
def create_angryMusic(db: Session, songItem:SongItem):
    angrymusic = AngryMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(angrymusic)
    db.commit()
    db.refresh(angrymusic)
    
def create_delightMusic(db: Session, songItem:SongItem):
    delightMusic = DelightMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(delightMusic)
    db.commit()
    db.refresh(delightMusic)
    
def create_calmMusic(db: Session, songItem:SongItem):
    calmMusic = CalmMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(calmMusic)
    db.commit()
    db.refresh(calmMusic)
    
def create_embrrasedMusic(db: Session, songItem:SongItem):
    embrrasedMusic = EmbrrasedMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(embrrasedMusic)
    db.commit()
    db.refresh(embrrasedMusic)
    
def create_anxietyMusic(db: Session, songItem:SongItem):
    anxietyMusic = AnxietyMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(anxietyMusic)
    db.commit()
    db.refresh(anxietyMusic)
    
def create_loveMusic(db: Session, songItem:SongItem):
    loveMusic = LoveMusic(spotipyId=songItem.spotifyId,
                        title = songItem.title,
                        angry = songItem.emotionList[6],
                        love = songItem.emotionList[5],
                        embrrased = songItem.emotionList[4],
                        anxiety = songItem.emotionList[3],
                        calm = songItem.emotionList[2],
                        delight = songItem.emotionList[1],
                        sad = songItem.emotionList[0])
    
    db.add(loveMusic)
    db.commit()
    db.refresh(loveMusic)

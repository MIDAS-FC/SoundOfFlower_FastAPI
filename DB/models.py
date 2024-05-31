from sqlalchemy import Column, TEXT, INT, BIGINT, DOUBLE
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Music(Base):
   __tablename__ = "music"

   spotify = Column(TEXT, primary_key=True)
   emotion_type = Column(TEXT, nullable=False)
   total_likes = Column(INT, nullable=False, default = 0)

#슬픔
class SadMusic(Base):
   __tablename__ = "sad_music"

   spotify = Column(TEXT, primary_key=True)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
   emotion_type = Column(TEXT, nullable=False, default="sad")

#기쁨
class DelightMusic(Base):
   __tablename__ = "delight_music"

   spotify = Column(TEXT, primary_key=True)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
   emotion_type = Column(TEXT, nullable=False, default="delight")

#사랑
class LoveMusic(Base):
   __tablename__ = "love_music"

   spotify = Column(TEXT, primary_key=True)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
   emotion_type = Column(TEXT, nullable=False, default="love")
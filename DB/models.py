from sqlalchemy import Column, TEXT, INT, BIGINT, DOUBLE
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

#슬픔
class SadMusic(Base):
   __tablename__ = "sad_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotifyId = Column(TEXT, nullable=False)
   title = Column(TEXT, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#기쁨
class DelightMusic(Base):
   __tablename__ = "delight_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotifyId = Column(TEXT, nullable=False)
   title = Column(TEXT, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#사랑
class LoveMusic(Base):
   __tablename__ = "love_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotifyId = Column(TEXT, nullable=False)
   title = Column(TEXT, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
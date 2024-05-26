from sqlalchemy import Column, TEXT, INT, BIGINT, DOUBLE
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

#슬픔
class SadMusic(Base):
   __tablename__ = "sad_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
   
#분노
class AngryMusic(Base):
   __tablename__ = "angry_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#기쁨
class DelightMusic(Base):
   __tablename__ = "delight_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#중립
class CalmMusic(Base):
   __tablename__ = "calm_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#당황
class EmbrrasedMusic(Base):
   __tablename__ = "embrrased_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
 
#불안  
class AnxietyMusic(Base):
   __tablename__ = "anxiety_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)

#사랑
class LoveMusic(Base):
   __tablename__ = "love_music"

   id = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
   spotipyId = Column(INT, nullable=False)
   title = Column(TEXT, nullable=False)
   angry = Column(DOUBLE, nullable=False)
   sad = Column(DOUBLE, nullable=False)
   delight = Column(DOUBLE, nullable=False)
   calm = Column(DOUBLE, nullable=False)
   embrrased = Column(DOUBLE, nullable=False)
   anxiety = Column(DOUBLE, nullable=False)
   love = Column(DOUBLE, nullable=False)
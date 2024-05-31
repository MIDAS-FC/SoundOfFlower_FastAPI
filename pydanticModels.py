from typing import List, Union
from pydantic import BaseModel

class SongItem(BaseModel):
    emotion: str
    emotionList: list[float]
    spotify: str
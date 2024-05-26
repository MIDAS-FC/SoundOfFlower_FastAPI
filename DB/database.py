from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

DB_URL = 'mysql+pymysql://root:1234@localhost:3306/sof_music'


class engineconn:

    # 클래스의 생성자 메서드. 객체가 생성될 때 호출되며, 데이터베이스 연결 엔진을 초기화한다.
    def __init__(self):
        self.engine = create_engine(DB_URL, pool_recycle = 500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn
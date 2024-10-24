import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

conn = pymysql.connect(host='127.0.0.1', user = 'root', password='Admin1234!!', db = 'shoe_project',charset = 'utf8')
#본인 db password 입력, database 이름 입력 
curs = conn.cursor(pymysql.cursors.DictCursor)

prod_img = pd.read_csv('./jh/prod_img.csv', index_col=False) #csv 파일 저장된 경로 
prod_img = prod_img.where((pd.notnull(prod_img)), 'N/A')
print(prod_img.head())

for index, row in prod_img.iterrows():
    tu = (row.prod_url, row.prod )   #db에 넣고자 하는 csv파일의 컬럼 입력 ex) rew.champion_id / 순서 상관 x
    curs.execute("""  INSERT IGNORE INTO shoe_project.shoe_detect_productimg      
    (img_path, prod_id) VALUES
    (%s, %s)""", tu)   # INSERT IGNORE INTO project.board_champion_counter  //  (본인database 이름).(database내 table 이름)   
    # ( ) VALUES -> csv를 넣을 database table의 컬럼 입력. 순서는 table 컬럼 순서 맞춰서
    # id 컬럼은 models.py에서 따로지정해주지 않은 autofield primary_key 입력해주지 않아도 됨.

conn.commit()
conn.close()
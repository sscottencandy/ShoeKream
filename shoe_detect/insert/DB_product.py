import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

conn = pymysql.connect(host='127.0.0.1', user = 'root', password='Admin1234!!', db = 'shoe_project',charset = 'utf8') #본인 db password 입력, database 이름 입력 
curs = conn.cursor(pymysql.cursors.DictCursor)

prod_info = pd.read_csv('./jh/product.csv', index_col=False) #csv 파일 저장된 경로 
prod_info = prod_info.where((pd.notnull(prod_info)), 'N/A')
print(prod_info.head())

for index, row in prod_info.iterrows():
    tu = (row.제품명, row.가격, row.발매가, row.발매일, row.브랜드, row.Kream번호 )   #db에 넣고자 하는 csv파일의 컬럼 입력 ex) rew.champion_id / 순서 상관 x
    curs.execute("""  INSERT IGNORE INTO shoe_project.shoe_detect_product      
    (name, price, original_price, date_release, brand, prod) VALUES
    (%s, %s, %s, %s, %s, %s)""", tu)   # INSERT IGNORE INTO project.board_champion_counter  //  (본인database 이름).(database내 table 이름)   
    # ( ) VALUES -> csv를 넣을 database table의 컬럼 입력. 순서는 table 컬럼 순서 맞춰서
    # id 컬럼은 models.py에서 따로지정해주지 않은 autofield primary_key 입력해주지 않아도 됨.

conn.commit()
conn.close()
import psycopg2
from datetime import datetime
import pandas as pd
import numpy as np

conn = psycopg2.connect(dbname='test_data', user='pauloarantes', host='/tmp')
c = conn.cursor()

c.execute(
    '''
SELECT
u.id,
u.last_sign_in_at::date,
u.created_at::date,
to_timestamp(CAST(bs.max_time as int))::date as last_session,
bs.num_sessions,
bs.total_artists_followed,
bs.total_artworks_favorited,
bs.total_artworks_shared,
bs.total_follows,
bs.total_favorites,
bs.last_favorited_artwork_date,
bs.last_followed_artist_date,
city.city,
gender.gender,
user_type.user_type,
os.os

FROM users u

JOIN (SELECT CAST(data->'properties'->>'user id' AS int) as user_id,
             COUNT(data->'properties'->>'user id') as num_sessions,
             MAX(data->'properties'->>'time') as max_time,
             MAX(data->'properties'->>'total artists followed') as total_artists_followed,
             MAX(data->'properties'->>'total artworks favorited')as total_artworks_favorited,
             MAX(data->'properties'->>'total artworks shared')as total_artworks_shared,
             MAX(data->'properties'->>'last favorited artwork date') as last_favorited_artwork_date,
             MAX(data->'properties'->>'last followed artist date') as last_followed_artist_date,
             MAX(data->'properties'->>'total follows') as total_follows,
             MAX(data->'properties'->>'total favorites') as total_favorites
        FROM mp_began_session
        GROUP BY data->'properties'->>'user id') bs
ON bs.user_id=u.id

FULL OUTER JOIN user_city city
ON u.id=city.user_id

FULL OUTER JOIN user_gender gender
ON u.id=gender.user_id

FULL OUTER JOIN user_os os
ON u.id=os.user_id

FULL OUTER JOIN user_type
ON u.id=user_type.user_id

'''
)

data = c.fetchall()

cols = ['id',
'last_sign_in_at',
'created_at',
'last_session',
'num_sessions',
'total_artists_followed',
'total_artworks_favorited',
'total_artworks_shared',
'total_follows',
'total_favorites',
'last_favorited_artwork_date',
'last_followed_artist_date',
'city',
'gender',
'user_type',
'os']

df = pd.DataFrame(data, columns=cols)

df.last_favorited_artwork_date = pd.to_datetime(df.last_favorited_artwork_date, errors='coerce').dt.date
df.last_followed_artist_date = pd.to_datetime(df.last_followed_artist_date, errors='coerce').dt.date

df.to_csv('dataset.csv')



c.execute('''
select user_id, total_pieces_purchased, total_spent, created_at from buyers;
''')

purch = c.fetchall()

df = pd.DataFrame(purch, columns=['user_id', 'total_pieces_purchased', 'total_spent', 'created_at'])
df.created_at = pd.to_datetime(df.created_at, errors='coerce').dt.date

df.to_csv('purchases.csv')


'''SUPER COOL:
select user_id, total_pieces_purchased, total_spent, created_at from buyers;
'''

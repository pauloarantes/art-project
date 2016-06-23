import psycopg2
from datetime import datetime

conn = psycopg2.connect(dbname='test_data', user='pauloarantes', host='/tmp')
c = conn.cursor()

c.execute(
'''
WITH cities AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'$city' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'$city' as city
                INTO user_city
                FROM mp_began_session mp
                JOIN user_cities_time c
                ON c.user_id=mp.data->'properties'->>'user id'
                AND c.max_time=mp.data->'properties'->>'time';
'''
)

c.execute(
'''
WITH genders AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'gender' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'gender' as gender
                INTO user_gender
                FROM mp_began_session mp
                JOIN genders
                ON genders.user_id=mp.data->'properties'->>'user id'
                AND genders.max_time=mp.data->'properties'->>'time';
'''
)

c.execute(
'''
WITH user_types AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'User Type' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'User Type' as user_type
                INTO user_type
                FROM mp_began_session mp
                JOIN user_types
                ON user_types.user_id=mp.data->'properties'->>'user id'
                AND user_types.max_time=mp.data->'properties'->>'time';
'''
)

c.execute(
'''
WITH oss AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'$os' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'$os' as os
                INTO user_os
                FROM mp_began_session mp
                JOIN oss
                ON oss.user_id=mp.data->'properties'->>'user id'
                AND oss.max_time=mp.data->'properties'->>'time';
'''
)

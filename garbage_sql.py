JOIN (SELECT user_id,
             SUM(total_pieces_purchased) as total_pieces_purchased,
             SUM(total_spent) as total_spent
        FROM buyers
        GROUP BY user_id) b
ON u.id=b.user_id

JOIN (WITH cities AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'$city' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'$city' as city
                FROM mp_began_session mp
                JOIN cities
                ON cities.user_id=mp.data->'properties'->>'user id'
                AND cities.max_time=mp.data->'properties'->>'time') city
ON city.user_id=u.id

JOIN (WITH genders AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'gender' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
        )
                SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
                             mp.data->'properties'->>'gender' as gender
                FROM mp_began_session mp
                JOIN genders
                ON genders.user_id=mp.data->'properties'->>'user id'
                AND genders.max_time=mp.data->'properties'->>'time') gender
ON gender.user_id=u.id



JOIN (SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'User Type' as user_type
        FROM mp_began_session mp
        JOIN (SELECT data->'properties'->>'user id' as user_id,
                     MAX(data->'properties'->>'time') as max_time
                FROM mp_began_session
                WHERE data->'properties'->>'User Type' IS NOT NULL
                GROUP BY data->'properties'->>'user id') user_types
        ON user_types.user_id=mp.data->'properties'->>'user id'
        WHERE mp.data->'properties'->>'time' = user_types.max_time) user_type
ON user_type.user_id=u.id

JOIN (SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'$os' as os
        FROM mp_began_session mp
        JOIN (SELECT data->'properties'->>'user id' as user_id,
                     MAX(data->'properties'->>'time') as max_time
                FROM mp_began_session
                WHERE data->'properties'->>'$os' IS NOT NULL
                GROUP BY data->'properties'->>'user id') oss
        ON oss.user_id=mp.data->'properties'->>'user id'
        WHERE mp.data->'properties'->>'time' = oss.max_time) os
ON os.user_id=u.id


LIMIT 30
;
'''

SELECT * INTO films_recent FROM films WHERE date_prod >= '2002-01-01';
'''
SELECT data->'properties'->>'user id' as user_id,
       MAX(data->'properties'->>'time') as max_time
INTO user_cities_time
FROM mp_began_session
WHERE data->'properties'->>'$city' IS NOT NULL
GROUP BY data->'properties'->>'user id'
;

JOIN (WITH cities AS(
        SELECT DISTINCT(data->'properties'->>'user id') as user_id,
                        data->'properties'->>'$city'
        FROM mp_began_session
        WHERE data->'properties'->>'$city' IS NOT NULL;

'''
-----
JOIN (SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'$city' as city
        FROM mp_began_session mp
        JOIN (SELECT data->'properties'->>'user id' as user_id,
                     MAX(data->'properties'->>'time') as max_time
                FROM mp_began_session
                WHERE data->'properties'->>'$city' IS NOT NULL
                GROUP BY data->'properties'->>'user id') cities
        ON cities.user_id=mp.data->'properties'->>'user id'
        WHERE mp.data->'properties'->>'time' = cities.max_time) city
ON city.user_id=u.id

JOIN (SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'gender' as gender
        FROM mp_began_session mp
        JOIN (SELECT data->'properties'->>'user id' as user_id,
                     MAX(data->'properties'->>'time') as max_time
                FROM mp_began_session
                WHERE data->'properties'->>'gender' IS NOT NULL
                GROUP BY data->'properties'->>'user id') genders
        ON genders.user_id=mp.data->'properties'->>'user id'
        WHERE mp.data->'properties'->>'time' = genders.max_time) gender
ON gender.user_id=u.id



----
SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'$city' as city
        FROM mp_began_session mp
        JOIN (SELECT data->'properties'->>'user id' as user_id,
                     MAX(data->'properties'->>'time') as max_time
                FROM mp_began_session
                WHERE data->'properties'->>'$city' IS NOT NULL
                GROUP BY data->'properties'->>'user id') cities
        ON cities.user_id=mp.data->'properties'->>'user id'
        WHERE mp.data->'properties'->>'time' = cities.max_time

WITH cities AS(
        SELECT data->'properties'->>'user id' as user_id,
               MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'$city' IS NOT NULL
        GROUP BY data->'properties'->>'user id'
)
SELECT  CAST(mp.data->'properties'->>'user id' AS int) as user_id,
              mp.data->'properties'->>'$city' as city
FROM mp_began_session mp
JOIN cities
ON cities.user_id=mp.data->'properties'->>'user id' AND cities.max_time=mp.data->'properties'->>'time'




------
JOIN (WITH temp_city AS (
        SELECT  data->'properties'->>'user id' as user_id,
                data->'properties'->>'$city' as city,
                DENSE_RANK() OVER(PARTITION BY 'user id' ORDER BY '$city' DESC) AS top
        FROM mp_began_session
        )
            SELECT DISTINCT (tc.user_id) as user_id, tc.city as city
            FROM temp_city tc
            WHERE tc.top = 1) city
ON city.user_id=u.id

JOIN (WITH temp_utype AS (
        SELECT  data->'properties'->>'user id' as user_id,
                data->'properties'->>'User Type' as utype,
                DENSE_RANK() OVER(PARTITION BY 'user id' ORDER BY 'User Type' DESC) AS top
        FROM mp_began_session
        )
            SELECT DISTINCT (CAST(tut.user_id as int)) as user_id, tut.utype as user_type
            FROM temp_utype tut
            WHERE tut.top = 1) user_type
ON user_type.user_id=u.id

JOIN (WITH temp_gender AS (
        SELECT  data->'properties'->>'user id' as user_id,
                data->'properties'->>'gender' as gender,
                DENSE_RANK() OVER(PARTITION BY 'user id' ORDER BY 'gender' DESC) AS top
        FROM mp_began_session
        )
            SELECT DISTINCT (CAST(tg.user_id as int)) as user_id, tg.gender as gender
            FROM temp_gender tg
            WHERE tg.top = 1) gender
ON gender.user_id=u.id

JOIN (WITH temp_os AS (
        SELECT  data->'properties'->>'user id' as user_id,
                data->'properties'->>'$os' as os,
                DENSE_RANK() OVER(PARTITION BY 'user id' ORDER BY '$os' DESC) AS top
        FROM mp_began_session
        )
            SELECT DISTINCT (CAST(tos.user_id as int)) as user_id, tos.os as os
            FROM temp_os tos
            WHERE tos.top = 1) os
ON os.user_id=u.id

-----


(SELECT  mp.data->'properties'->>'user id' as user_id,
        mp.data->'properties'->>'$city' as city
FROM mp_began_session mp
JOIN (SELECT data->'properties'->>'user id' as user_id, MAX(data->'properties'->>'time') as max_time
        FROM mp_began_session
        WHERE data->'properties'->>'$city' IS NOT NULL
        GROUP BY data->'properties'->>'user id') cities
ON cities.user_id=mp.data->'properties'->>'user id'
WHERE mp.data->'properties'->>'time' = cities.max_time) city


# data->'properties'->>'city',
# data->'properties'->>'User Type',
# data->'properties'->>'gender',
#
# SELECT data->'properties'->>'user id', COUNT(data->'properties'->>'user id'), MAX(data->'properties'->>'total artists followed')
# FROM mp_began_session
# GROUP BY data->'properties'->>'user id' LIMIT 10;
#
# where CAST(data->'properties'->>'total artists followed' AS int)>0
#
# )

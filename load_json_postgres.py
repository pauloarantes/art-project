import os
import sys
import json
import psycopg2 as pg
import psycopg2.extras


# data_dir
filename = '/Users/pauloarantes/Drive/galvanize/_capstone/art-project/all.json'

# Connected to DB using credentials
conn = pg.connect(dbname='test_data', user='pauloarantes', host='/tmp')
c = conn.cursor()

# Drop tables with old data
c.execute("DROP TABLE mp_purchases;")
c.execute("DROP TABLE mp_began_session;")

# Creates SQL table for new data entries
c.execute('''CREATE TABLE mp_purchases (id INTEGER PRIMARY KEY, data JSONB NOT NULL);''')
c.execute('''CREATE TABLE mp_began_session (id INTEGER PRIMARY KEY, data JSONB NOT NULL);''')

# Population SQL table with JSONB objects from MixPanel event based data
with open(filename) as js_file:
    for line_count, js_line in enumerate(js_file):

        js_type = json.loads(js_line)['event']

        if js_type == '_V2 User began session':
            c.execute("""INSERT INTO mp_began_session(id, data) VALUES (%s, %s);""", (line_count, js_line))
        elif js_type == '_V2 User completed purchase':
            c.execute("""INSERT INTO mp_purchases(id, data) VALUES (%s, %s);""", (line_count, js_line))
        else: # no js_typ match, FAULT!
            pass

conn.commit()
conn.close()

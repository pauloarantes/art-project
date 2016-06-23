#!/usr/bin/env python
# Found this script from http://blog.endpoint.com/2016/03/loading-json-files-into-postgresql-95.html
#

import os
import sys
import logging

try:
    import psycopg2 as pg
    import psycopg2.extras
except:
    print "Install psycopg2"
    exit(123)

import json

# data_dir = 'test_data'
filename = '/Users/pauloarantes/Drive/galvanize/_capstone/art-project/all.json'

conn = pg.connect(dbname='test_data', user='pauloarantes', host='/tmp')
c = conn.cursor()

# --- created SQL table ---

c.execute('''CREATE TABLE mp_purchases (id INTEGER PRIMARY KEY, data JSONB NOT NULL);''')
c.execute('''CREATE TABLE mp_began_session (id INTEGER PRIMARY KEY, data JSONB NOT NULL);''')

with open(filename) as js_file:

    for line_count, js_line in enumerate(js_file):
        # print line_count, js_line
        js_type = json.loads(js_line)['event']

        if js_type == '_V2 User began session':
            c.execute("""INSERT INTO mp_began_session(id, data) VALUES (%s, %s);""", (line_count, js_line))
        elif js_type == '_V2 User completed purchase':
            c.execute("""INSERT INTO mp_purchases(id, data) VALUES (%s, %s);""", (line_count, js_line))
        else: # no js_typ match, FAULT!
            pass
conn.commit() # <--- this line saved Dean's 'life'!

conn.commit()
conn.close()

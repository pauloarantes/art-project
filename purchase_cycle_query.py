import psycopg2
from datetime import datetime
import pandas as pd
import numpy as np

conn = psycopg2.connect(dbname='test_data', user='pauloarantes', host='/tmp')
c = conn.cursor()

c.execute('''
SELECT user_id, total_pieces_purchased, total_spent, created_at
FROM buyers
ORDER BY 1;
''')

data = c.fetchall()

df = pd.DataFrame(data, columns=['id', 'pieces', 'spent', 'when'])

df.to_csv('purchase_cycles.csv')

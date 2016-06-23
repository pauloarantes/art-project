import pandas as pd
import numpy as np
import json

pd.options.display.max_columns=999

def load_json_data(filename):
    with open(filename, 'rb') as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    data_df = pd.read_json(data_json_str)
    data_df.properties = data_df.properties.apply(json.dumps)
    return data_df

def subset_event(df, event_name=None):
    """
    Available events: zip(list(data_df.event.value_counts().index), data_df.event.value_counts().values)
    """
    if event_name:
        df = df[df.event == event_name]
    event_features = df.properties.values
    event_features = map(lambda x: x.rstrip(), event_features)
    data_json_str_event_features = "[" + ','.join(event_features) + "]"
    event_df = pd.read_json(data_json_str_event_features)
    return event_df


if __name__ == '__main__':
    data_df = load_json_data('test_mp.json')
    purchases_df = subset_event(data_df)


"""
db:
purchase_items from purchases
buyer_id to user_id or not
scope :successful, -> { where(purchase_state: ["billing_successful", "billed"])}

wc -l all.json

pg_restore --create --clean --if-exists -Fd -d arttwo50_dev -j8 --no-owner -Ustephen db/prod_dump/
pg_restore --create --clean --if-exists -Fd -j8 --no-owner -Upostgres -d test_data /Users/pauloarantes/Drive/galvanize/_capstone/art-project/VangoDB

"""

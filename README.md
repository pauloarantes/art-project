# art-project
Capstone project for Galvanize

1) Run mixpanel_export.py script to get jsons from Mixpanel events (specify desired dates)

2) Restore the database dump with the following command:
    CREATE DATABASE test_data;
    psql -d test_data -f /Users/pauloarantes/Drive/galvanize/_capstone/art-project/prod_dump;

    If it doesn't work, try:
    pg_restore --create --clean --if-exists -Fd -j8 --no-owner -Upauloarantes -d test_data /Users/pauloarantes/Drive/galvanize/_capstone/art-project/prod_dump;

3) Run load_json_postgres to process and load Mixpanel data into the database as JSONB objects

4) Run prep-queries.py to generate assistant tables on the database to improve querying process time and memory

5) Run queries.py to query from db and export 2 csvs (purchases.csv and dataset.csv)

6) Run purchase_cycle_query.py to query from db and export 1 csv (purchase_cycle.csv)

7)


Next steps:
*) profit curve
*) datasetEDA.py -> generate pretty plots, including all purchase cycle analysis
*) Visualization of accesses by lat lon

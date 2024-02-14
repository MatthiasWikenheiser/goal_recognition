import sqlite3 as db
import pandas as pd
import numpy as np

db_path = "/home/mwiubuntu/Seminararbeit/db_results/test_goal_recognition.db"

def accuracy(model_type, hash_code_model, hash_code_action, rl_type=0, iterations=None,
             station=None, log_file=None,
             file_tuples = None,
             multiclass = True):
    if not file_tuples is None and (not station is None or log_file is None):
        print("please only use parameter file_tuples or station in combination with file_tuples")
    query = f"""SELECT correct_prediction
               FROM model_grid_observed
               WHERE model_type = '{model_type}'
                     AND hash_code_model ='{hash_code_model}'
	                 AND hash_code_action = '{hash_code_action}'
	                 AND rl_type = {rl_type}
	                 AND label IS NOT NULL"""
    if not multiclass:
        query += "\n\t\t\t\t\t AND predicted_goals_no < 2"

    if iterations is None:
        query += "\n\t\t\t\t\t AND iterations IS NULL"
    else:
        query += f"\n\t\t\t\t\t AND iterations = {iterations}"

    if not station is None:
        query += f"\n\t\t\t\t\t AND station = '{station}'"

    if not log_file is None:
        query += f"\n\t\t\t\t\t AND log_file = '{log_file}'"
    print(query)

    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return np.mean(df)


if __name__ == '__main__':
    model_type = "prap_model"
    hash_code_model = '52f3fe1ade9258da2452dcadb3c9c8836828d95be112a31f36f38f3c'
    hash_code_action = '222b41d94ac651c514738913617e0f3fcc8f57ebf623546630e8540b'
    station = 'Session1-StationA'
    log_file = '2_log_Salmonellosis.csv'
    print(accuracy(model_type, hash_code_model, hash_code_action, multiclass=True, station=station, log_file=log_file))



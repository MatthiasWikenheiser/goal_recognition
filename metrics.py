import sqlite3 as db
import pandas as pd
import numpy as np

db_path = "/home/mwiubuntu/Seminararbeit/db_results/goal_recognition.db"

def _get_hash_code_models(model_type):
    query = f"""SELECT DISTINCT(hash_code_model) 
                FROM model_grid_observed
                WHERE model_type = '{model_type}'"""
    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return list(df["hash_code_model"])

def _get_hash_code_actions(model_type, hash_code_model):
    query = f"""SELECT DISTINCT(hash_code_action) 
                FROM model_grid_observed 
                WHERE model_type = '{model_type}'
                    AND hash_code_model = '{hash_code_model}'"""
    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return list(df["hash_code_action"])

def _get_hash_code_model_name(hash_code_model):
    query = f"""SELECT model_name FROM model WHERE hash_code_model = '{hash_code_model}'"""
    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return df["model_name"].iloc[0]

def _get_model_types():
    query = f"""SELECT DISTINCT(model_type) FROM model_grid_observed"""
    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return list(df["model_type"])

def accuracy(model_type, hash_code_model, hash_code_action, rl_type=0, iterations=None,
             station=None, log_file=None,
             file_tuples = None,
             multiclass = True):
    """under construction"""
    if not file_tuples is None and (not station is None or log_file is None):
        print("please only use parameter file_tuples or station in combination with file_tuples")
        return None
    query = f"""SELECT correct_prediction
               FROM model_grid_observed
               WHERE model_type = '{model_type}'
                     AND hash_code_model ='{hash_code_model}'
	                 AND hash_code_action = '{hash_code_action}'
	                 AND rl_type = {rl_type}
	                 AND label IS NOT NULL
                     AND total_goals_no > 1"""
    if not multiclass:
        query += "\n\t\t\t\t\t AND predicted_goals_no < 2"

    if iterations is None:
        query += "\n\t\t\t\t\t AND iterations IS NULL"
    else:
        query += f"\n\t\t\t\t\t AND iterations = {iterations}"

    if not station is None or not log_file is None:
        if len(station) != len(log_file):
            print("ERROR: station and log_files of different length")
            return None
        else:
            query += f"\n\t\t\t\t\t AND ("
            i=0
            while i < len(station):
                if i != 0:
                    query += f"\n\t\t\t\t\t\t OR"
                query += f" (station = '{station[i]}' AND log_file = '{log_file[i]}')"
                i+=1
        query += f"\n\t\t\t\t\t )"
        db_gr = db.connect(db_path)
        df = pd.read_sql_query(query, db_gr)
        db_gr.close()
        return df["correct_prediction"].mean()
    else:
        db_gr = db.connect(db_path)
        df = pd.read_sql_query(query, db_gr)
        db_gr.close()
        return df["correct_prediction"].mean()

def collect_accuracy():
    """under construction"""
    results = pd.DataFrame()
    for model_type in _get_model_types():
        for hash_code_model in _get_hash_code_models(model_type):
            model_name = _get_hash_code_model_name(hash_code_model)
            for hash_code_action in _get_hash_code_actions(model_type=model_type, hash_code_model=hash_code_model):
                accur = accuracy(model_type, hash_code_model, hash_code_action, multiclass=True)
                result_entry = pd.DataFrame({"model_type": [model_type],
                                             "hash_code_model": [hash_code_model],
                                             "model_name": [model_name],
                                             "hash_code_action": [hash_code_action],
                                             "accuracy": [accur]})
                results = pd.concat([results, result_entry])
    results = results.reset_index().iloc[:, 1:]
    return results

def convergence_rate(model_type, hash_code_model, hash_code_action, rl_type=0, iterations=None,
                     station=None, log_file=None):
    result = _convergence_rate_df(model_type, hash_code_model, hash_code_action, rl_type=rl_type,
                             iterations=iterations,
                             station=station, log_file=log_file)
    return result["correct_prediction"].mean()

def _convergence_rate_df(model_type, hash_code_model, hash_code_action, rl_type=0, iterations=None,
                     station=None, log_file=None):
    identifier = ["model_type", "hash_code_model", "hash_code_action", "station", "log_file", "rl_type"]
    if not iterations is None:
        identifier = identifier + ["iterations"]
    result = _get_model_grid_observed(model_type, hash_code_model, hash_code_action, rl_type=rl_type,
                                      iterations=iterations,
                                      station=station, log_file=log_file)
    last_goal_obs = result.groupby(identifier + ["label"], as_index=False)["observed_action_no"].max()
    last_goal_obs = last_goal_obs.merge(result[identifier + ["label", "observed_action_no", "correct_prediction"]],
                                        on=identifier + ["label", "observed_action_no"], how="left")
    return last_goal_obs


def _get_model_grid_observed(model_type, hash_code_model, hash_code_action, rl_type, iterations,
                     station, log_file):
    """under construction"""
    if not((station is None and log_file is None) or (not station is None and not log_file is None)):
        print("please choose station AND log_file, otherwise set BOTH to None")
        return None
    query = f"""
               SELECT *
               FROM model_grid_observed
               WHERE model_type = '{model_type}'
                 AND hash_code_model ='{hash_code_model}'
                 AND hash_code_action = '{hash_code_action}'
                 AND rl_type = {rl_type}
                 AND label IS NOT NULL
                 AND total_goals_no > 1"""
    if iterations is None:
        query += "\n\t\t\t\t AND iterations IS NULL"
    else:
        query += f"\n\t\t\t\t AND iterations = {iterations}"

    if not station is None:
        query += f"\n\t\t\t\t AND station = '{station}'"

    if not log_file is None:
        query += f"\n\t\t\t\t AND log_file = '{log_file}'"

    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()

    return df

def collect_convergence_rate():
    """under construction"""
    results = pd.DataFrame()
    for model_type in _get_model_types():
        for hash_code_model in _get_hash_code_models(model_type=model_type):
            model_name = _get_hash_code_model_name(hash_code_model)
            for hash_code_action in _get_hash_code_actions(model_type=model_type, hash_code_model=hash_code_model):
                cr = convergence_rate(model_type,hash_code_model, hash_code_action)
                result_entry = pd.DataFrame({"model_type": [model_type],
                                             "hash_code_model": [hash_code_model],
                                             "model_name": [model_name],
                                             "hash_code_action": [hash_code_action],
                                             "convergence_rate": [cr]})
                results = pd.concat([results, result_entry])
    results = results.reset_index().iloc[:, 1:]
    return results

def cross_validation(model_type, hash_code_model, hash_code_action, rl_type = 0, iterations = None,
                     k=5, multiclass=True):
    query = f"""SELECT station, log_file 
                FROM model_grid_observed 
                WHERE model_type = '{model_type}'
                    AND hash_code_model = '{hash_code_model}'
                    AND hash_code_action = '{hash_code_action}'
                    AND observed_action_no = 1"""
    db_gr = db.connect(db_path)
    station_files = pd.read_sql_query(query, db_gr)
    db_gr.close()
    split_size = (1/k)*len(station_files)
    if split_size % 1 != 0:
        split_size = int(split_size) + 1
    i = 0
    acc_list = []
    while i < k:
        left_bound = i*split_size
        if i == k-1:
            split_df = station_files.iloc[left_bound:]
        else:
            right_bound = (i+1) * split_size
            split_df =station_files.iloc[left_bound:right_bound]
        stations = list(split_df["station"])
        log_files = list(split_df["log_file"])
        acc_list.append(accuracy(model_type=model_type, hash_code_model=hash_code_model,
                 hash_code_action=hash_code_action, rl_type=rl_type, iterations=iterations,
                 station=stations, log_file=log_files, multiclass=multiclass))
        i+=1
    return acc_list, np.mean(acc_list)




def collect_convergence_point(exclude_null_predictions=True):
    """under construction"""
    results = pd.DataFrame()
    for model_type in _get_model_types():
        for hash_code_model in _get_hash_code_models(model_type=model_type):
            model_name = _get_hash_code_model_name(hash_code_model=hash_code_model)
            for hash_code_action in _get_hash_code_actions(model_type=model_type,hash_code_model=hash_code_model):
                print(model_type, hash_code_model, model_name, hash_code_action, exclude_null_predictions)
                cp = convergence_point(model_type, hash_code_model, hash_code_action,
                                       exclude_null_predictions=exclude_null_predictions)
                result_entry = pd.DataFrame({"model_type": [model_type],
                                             "hash_code_model": [hash_code_model],
                                             "model_name": [model_name],
                                             "hash_code_action": [hash_code_action],
                                             "convergence_point": [cp]})
                results = pd.concat([results, result_entry])
    results = results.reset_index().iloc[:, 1:]
    return results

def convergence_point(model_type, hash_code_model, hash_code_action, rl_type=0, iterations=None,
                      station=None, log_file=None, exclude_null_predictions = True):
    null_values_exist = False
    identifier = ["model_type", "hash_code_model", "hash_code_action", "station", "log_file", "rl_type"]
    if not iterations is None:
        identifier = identifier + ["iterations"]
    if not((station is None and log_file is None) or (not station is None and not log_file is None)):
        print("please choose station AND log_file, otherwise set BOTH to None")
        return None
    convergence_rate_df = _convergence_rate_df(model_type, hash_code_model, hash_code_action, rl_type=rl_type,
                                                 iterations=iterations, station=station, log_file=log_file)
    convergence_rate_df = convergence_rate_df[convergence_rate_df["correct_prediction"] == 1]
    m = len(convergence_rate_df)
    numerator = 0
    entities = convergence_rate_df.groupby(identifier, as_index=False).count()[identifier]
    for i in range(len(entities)):
        station_i = entities.loc[i,"station"]
        log_file_i = entities.loc[i,"log_file"]
        convergence_rate_df_i = _convergence_rate_df(model_type, hash_code_model, hash_code_action, rl_type=rl_type,
                                                   iterations=iterations, station=station_i, log_file=log_file_i)
        convergence_rate_df_i.sort_values("observed_action_no", ascending=True, inplace=True)
        convergence_rate_df_i = convergence_rate_df_i[convergence_rate_df_i["correct_prediction"] == 1]
        convergence_rate_df_i = convergence_rate_df_i.reset_index().iloc[:, 1:]
        converged_goals = list(convergence_rate_df_i["label"].unique())
        log_file_results = _get_model_grid_observed(model_type, hash_code_model, hash_code_action, rl_type=rl_type,
                                                    iterations=iterations, station=station_i, log_file=log_file_i)
        log_file_results = log_file_results[log_file_results["label"].isin(converged_goals)]
        log_file_results = log_file_results.reset_index().iloc[:, 1:]
        for goal in converged_goals:
            goal_observations = log_file_results[log_file_results["label"] == goal]
            goal_observations = goal_observations.reset_index().iloc[:, 1:]
            n_i = len(goal_observations)
            if exclude_null_predictions:
                keep = False
                for j in range(len(goal_observations)):
                    correct_prediction_j = goal_observations.loc[j,"correct_prediction"]
                    if correct_prediction_j == 1 and not keep:
                        k_i = j+1
                        keep = True
                    elif correct_prediction_j == 0:
                        keep = False
            else:
                not_keep = False
                for j in range(len(goal_observations)):
                    correct_prediction_j = goal_observations.loc[j, "correct_prediction"]
                    predicted_goals_no = goal_observations.loc[j, "predicted_goals_no"]
                    if correct_prediction_j == 1 and not not_keep:
                        k_i = j+1
                        not_keep = True
                    elif correct_prediction_j == 0 and predicted_goals_no != 0:
                        not_keep = False
            numerator += (k_i/n_i)
    return numerator/m


if __name__ == '__main__':
    model_7_hash = '0af2422953491e235481a3b7e0a10b74afc640356989b94d627359d1'
    #model_configs = ['495afca3d199dd8d66b44b1c5e414f225a19d42c9a540eabdcfec02e',
                     #'756b69e2c687b5c94fd2f2bc8214c53545b743fb43b4a2b0db86637a',
                     #'9ba2b59e1b25711ae06d73ddd96bb4d642744cdb5bc0ac291131329f',
                     #'b9b27945d52fbb94efbd91229da494b052aaaf36cd0026c4daf9bbfd',
                     #'d333764c7affc1cf43dd03e9bfc62876569666fbdabfeb66cc4a1f09'
                     #]
    #for model_config in model_configs:
        #print(model_config, cross_validation(model_type="gm_prap_model",
                               #hash_code_model=model_7_hash,
                               #hash_code_action=model_config))




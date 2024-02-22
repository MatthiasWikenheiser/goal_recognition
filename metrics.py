import sqlite3 as db
import pandas as pd

db_path = "/home/mwiubuntu/Seminararbeit/db_results/test_goal_recognition.db"

def _get_hash_code_models():
    query = f"""SELECT DISTINCT(hash_code_model) FROM model_grid_observed"""
    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return list(df["hash_code_model"])

def _get_hash_code_actions(hash_code_model):
    query = f"""SELECT DISTINCT(hash_code_action) 
                FROM model_grid_observed 
                WHERE hash_code_model = '{hash_code_model}'"""
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

    if not station is None:
        query += f"\n\t\t\t\t\t AND station = '{station}'"

    if not log_file is None:
        query += f"\n\t\t\t\t\t AND log_file = '{log_file}'"
    #print(query)

    db_gr = db.connect(db_path)
    df = pd.read_sql_query(query, db_gr)
    db_gr.close()
    return df["correct_prediction"].mean()

def collect_accuracy():
    """under construction"""
    results = pd.DataFrame()
    for model_type in _get_model_types():
        for hash_code_model in _get_hash_code_models():
            model_name = _get_hash_code_model_name(hash_code_model)
            for hash_code_action in _get_hash_code_actions(hash_code_model):
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
        for hash_code_model in _get_hash_code_models():
            model_name = _get_hash_code_model_name(hash_code_model)
            for hash_code_action in _get_hash_code_actions(hash_code_model):
                cr = convergence_rate(model_type,hash_code_model, hash_code_action)
                result_entry = pd.DataFrame({"model_type": [model_type],
                                             "hash_code_model": [hash_code_model],
                                             "model_name": [model_name],
                                             "hash_code_action": [hash_code_action],
                                             "convergence_rate": [cr]})
                results = pd.concat([results, result_entry])
    results = results.reset_index().iloc[:, 1:]
    return results

def collect_convergence_point(exclude_null_predictions=True):
    """under construction"""
    results = pd.DataFrame()
    for model_type in _get_model_types():
        for hash_code_model in _get_hash_code_models():
            model_name = _get_hash_code_model_name(hash_code_model)
            for hash_code_action in _get_hash_code_actions(hash_code_model):
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
            try:
                numerator += (k_i/n_i)
            except:
                null_values_exist = True
    if not null_values_exist:
        try:
            return numerator/m
        except:
            return None
    else:
        return None

if __name__ == '__main__':
    model_type = "prap_model"
    hash_code_model = '52f3fe1ade9258da2452dcadb3c9c8836828d95be112a31f36f38f3c'
    hash_code_action = '222b41d94ac651c514738913617e0f3fcc8f57ebf623546630e8540b'
    station = None#'Session1-StationC'
    log_file = None#'1_log_Salmonellosis.csv'
    #print(accuracy(model_type, hash_code_model, hash_code_action, multiclass=True, station=station, log_file=log_file))
    #results = collect_accuracy()
    #results.sort_values("accuracy", ascending= False, inplace = True)
    #print(convergence_rate(model_type="gm_model", hash_code_model=hash_code_model, hash_code_action=hash_code_action,
                     #station=station, log_file=log_file))
    #print(convergence_rate(model_type="prap_model", hash_code_model=hash_code_model, hash_code_action=hash_code_action,
                        #station=station, log_file=log_file))
    #convergence_rate_df = _convergence_rate_df(model_type="prap_model", hash_code_model=hash_code_model, hash_code_action=hash_code_action,
                     #station=station, log_file=log_file, rl_type=0, iterations = None)

    #convergence_collect = collect_convergence_rate()
    #convergence_collect.sort_values("convergence_rate", ascending=False, inplace=True)

    #cp = convergence_point(model_type=model_type, hash_code_model=hash_code_model,
                           #hash_code_action=hash_code_action, station=station, log_file=log_file,
                           #exclude_null_predictions=False)
    print("exclude_true")
    cp_collect_exclude_true = collect_convergence_point(exclude_null_predictions=True)
    cp_collect_exclude_true.sort_values("convergence_point", ascending=True, inplace=True)
    #print("exclude_false")
    #cp_collect_exclude_false = collect_convergence_point(exclude_null_predictions=False)
    #cp_collect_exclude_false.sort_values("convergence_point", ascending=True, inplace=True)




import pandas as pd
def _multiprocess_df(tuple_array):
    #must be outside gs_random_generator-class due to authentification string problem while multiprocessing
    array = tuple_array[0]
    list_grid_items = tuple_array[1]
    offset = len(tuple_array[1])
    df_baseline = tuple_array[2]
    dict_actions = tuple_array[3]
    array_idx_strt = 0
    array_idx_end = array_idx_strt + offset
    row_in_array = 0
    result_df = pd.DataFrame()
    for row in range(int(len(array)/offset)):
        conf = array[array_idx_strt:array_idx_end]
        array_idx_strt += offset
        array_idx_end += offset
        new_df = pd.DataFrame(columns = df_baseline.columns)
        for c in new_df.columns[3:]:
            if c in [col[0] for col in list_grid_items]:
                pos = [key for key in dict_actions.keys() if c == dict_actions[key][0]][0]
                new_df.loc[0,c] = conf[pos]
            else:
                new_df.loc[0,c] = df_baseline.loc[0,c]
        result_df = pd.concat([result_df,new_df])
    return result_df
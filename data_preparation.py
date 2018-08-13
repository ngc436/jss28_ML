__author__ = 'Maria Khodorchenko'

import pandas as pd
import os
import re

COLUMN_NAMES = ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
                'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
                'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
                'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
                'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']


def create_df(dir_name):
    list_of_frames = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            print(subdir[6:8])
            df = pd.read_csv(os.path.join(subdir, file), header=None, names=COLUMN_NAMES)
            df['Action'] = subdir[6:8]
            df['Subject'] = subdir[10:11]
            list_of_frames.append(df)
    result = pd.concat(list_of_frames)
    result.apply(pd.to_numeric)
    result.names = COLUMN_NAMES
    return result


def save_to_csv(df, fname):
    df.to_csv(fname)


def main():
    new_df = create_df("data")
    save_to_csv(new_df, "prepared.csv")


if __name__ == '__main__':
    main()

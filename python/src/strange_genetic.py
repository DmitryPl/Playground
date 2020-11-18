import random
import string

import pandas as pd

BLOCK_SIZE = 10
DATAFRAME_SIZE = 100


def build_block(size):
    return ''.join(random.choice(string.ascii_letters) for _ in range(size))


def fit(chars: str) -> int:
    num = 0
    for char in 'abcdefgABCDEFG':
        num += chars.count(char)
    return num


def generate_best(series: pd.Series, new_size: int) -> pd.Series:
    size, obj_size = len(series), len(series.values[0])
    tmp = [''] * new_size
    for i in range(new_size):
        hm = ''
        for _ in range(obj_size):
            idx, obj = random.randint(0, size - 1), random.randint(0, obj_size - 1)
            hm += series.values[idx][obj]
        tmp[i] = hm
    return pd.Series(tmp)


def genetic(data: pd.DataFrame, iterations: int, best: float, cut: float) -> pd.DataFrame:
    size = len(data)
    best_size, cut_size = int(size * best), int(size * cut)
    for _ in range(iterations):
        data['fit'] = data['data'].apply(fit)
        data = data.sort_values('fit', ascending=False)[:size - cut_size].drop(columns=['fit'])
        data['data'] = data['data'].append(generate_best(data['data'][:best_size], cut_size), ignore_index=True)
    return data


_data = pd.DataFrame()
_data['data'] = pd.Series([build_block(BLOCK_SIZE) for _ in range(DATAFRAME_SIZE)])
_data = genetic(_data, 5, 0.1, 0.2)
_data['fit'] = _data['data'].apply(fit)

print(_data['fit'].sum())

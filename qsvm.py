import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('datasets/vlds.csv', index_col=0)
    df.count()


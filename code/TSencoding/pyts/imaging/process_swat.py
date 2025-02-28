import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(-1, 1)) # scale training data to [0,1] range
    normalizer.fit(train)
    train_ret = normalizer.transform(train)
    normalizer.fit(test)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret

def z_norm(train, test):
    scaler = StandardScaler()
    train_ret = scaler.fit_transform(train)
    test_ret = scaler.fit_transform(test)
    
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    
    train_re = normalizer.fit_transform(train_ret)
    test_re = normalizer.fit_transform(test_ret)
    
    return train_re, test_re

# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)

    d_labels = np.round(np.max(d_labels, axis=1))


    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()




def main():
    test = pd.read_csv('./SWaT_Dataset_Attack_v0.csv', index_col=0)
    train = pd.read_csv('./SWaT_Dataset_Normal_v0.csv', index_col=0)

    test = test.iloc[:, 1:]
    train = train.iloc[:, 1:]
    
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    train_labels = train.attack
    test_labels = test.attack

    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    x_train, x_test = norm(train.values, test.values)
#    x_train, x_test = z_norm(train.values, test.values)


    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]

        
    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)

    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels


    train_df.to_csv('./train_process.csv')
    test_df.to_csv('./test_process.csv')

    f = open('./list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()


import math
import numpy as np 
import pandas_datareader as data_reader

def sigmoid(x):
    return 1/(1 + math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "-R$ {:.2f}".format(abs(n))
    else:
        return "R$ {:.2f}".format(abs(n))
    

def dataset_loader(stock_name):
    dataset = data_reader.DataReader(stock_name, data_source='yahoo')

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    close = dataset['Close']

    return close


def state_creator(data, timestep, window_size):
        starting_id = timestep - window_size + 1

        if starting_id >=0:
            print("TESTE: ", data[starting_id:timestep + 1])
            windowed_data = data[starting_id:timestep + 1]
        else:
            windowed_data = starting_id * [data[0]] + list(data[0:timestep + 1])

        state = []

        for x in range(window_size - 1):
            state.append(sigmoid(windowed_data[x+1] - windowed_data[x]))

            print("/nTESTE2: ", state)
            print("TYPE: ", type(windowed_data))
        
        return np.array([state]), windowed_data
            
            
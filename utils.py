import math
import numpy as np 
import pandas_datareader as data_reader
import argparse

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
            windowed_data = data[starting_id:timestep + 1]
        else:
            windowed_data = -starting_id * [data[0]] + list(data[0:timestep + 1])

        state = []

        for x in range(window_size - 1):
            state.append(sigmoid(windowed_data[x+1] - windowed_data[x]))

        
        return np.array([state]), windowed_data

def parse_config_file(config_file):
    file = open(config_file, 'r')

    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) !=0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key] = value.lstrip()
    blocks.append(block)
    return blocks

def arg_parse():
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument("--stock_name", dest="stock_name", help="Apelido da empresa na NASDAQ", default="MSFT")
    return parser.parse_args()

    
                
            
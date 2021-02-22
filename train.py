from model import AITRADE
from tqdm import tqdm_notebook, tqdm
from utils import *
import matplotlib.pyplot as plt 
import os
import warnings

if not os.sys.warnoptions:
    warnings.simplefilter("ignore")

hiper = parse_config_file('./config/hiperparameters.cfg')
arg = arg_parse()

stock_name = str(arg.stock_name)
window_size = int(hiper[0]['window_size'])
epochs = int(hiper[0]['episodes'])
batch_size = int(hiper[0]['batch_size'])
state_size =  int(hiper[0]['state_size'])

data = dataset_loader(stock_name)
data_samples = len(data) - 1
directory = './weights'

trader = AITRADE(state_size)

print(trader.model.summary())

for episode in range(1, epochs + 1):
    print(f'Etapa: {episode} de {epochs}. ')

    state = state_creator(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []
    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0
        
        if action == 1:
            trader.inventory.append(data[t])
            print("AI Trader sold: ", stocks_price_format(data[t]))
        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)

            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price

            print("AI Trader bought: ", stocks_price_format(data[t]),
                "Profit: " + stocks_price_format(data[t] - buy_price))
        
        if t == data_samples - 1:
            done = True
        else:
            done = False

            trader.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("###############################################")
        print(f'Total profit: {total_profit}')
        print("###############################################")

    if len(trader.memory) > batch_size:
        trader.batch_train(batch_size)

if episode % 10 == 0:

    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except Exception as e:
            raise e
    
    trader.model.save("weights/ai_trader_{}.h5".format(episode))
            

       



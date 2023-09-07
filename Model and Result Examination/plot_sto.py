import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tabulate import tabulate
from PIL import Image
from Model_comparison_functions import *

Add_on_path = "\\Results\\Sto 2 Sample\\2\\"
current_directory = "C:\\Users\\Daniel  BV\\Desktop\\Thesis"  #
choose_id = 'f1_d2_upd41_t42'


# check changes
model = 'Sto'
results = import_test_case(current_directory, Add_on_path, choose_id,models=model)
bid_result = results[model]['Bid']

# Prices

prices = ['f_FD2_up_tw_input','f_FD2_dn_tw_input','f_DA_tw_input','f_FD1_up_tw_input','f_FD1_dn_tw_input']
accep = []
bids = np.array([['b_FD2_up','b_FD2_dn'],['b_DA_up_all','b_DA_dn_all'],['b_FD1_up_all','b_FD1_dn_all']])
f_FD2_up_tw_input=  bid_result['f_FD2_up_tw_input']
f_FD2_dn_tw_input = bid_result['f_FD2_dn_tw_input']
f_DA_tw_input = bid_result['f_DA_tw_input']
f_FD1_up_tw_input = bid_result['f_FD1_up_tw_input']
f_FD1_dn_tw_input = bid_result['f_FD1_dn_tw_input']

b_FD2_up_all = bid_result['b_FD2_up']
b_FD2_dn_all = bid_result['b_FD2_dn']
b_DA_up_all  = bid_result['b_DA_up_all']
b_DA_dn_all  = bid_result['b_DA_dn_all']
b_FD1_up_all = bid_result['b_FD1_up_all']
b_FD1_dn_all = bid_result['b_FD1_dn_all']


for i in range(0,len(prices)):
    print(np.shape(bid_result[prices[i]]))






# bid_result['f_FD2_dn_tw_input']
# bid_result['f_DA_tw_input']
# bid_result['f_FD1_up_tw_input']
# bid_result['f_FD1_dn_tw_input']

sample_size = len(bid_result["b_FD1_dn_all"][:,0,0])
fig, ax = plt.subplots(2,3)

for i in range(0,3):
    for j in range(0,2):
        print(bids[i,j])
        print(np.shape(bid_result[bids[i,j]]))
        if "1" in bids[i,j]:
            for p1 in range(0,sample_size):
                for p2 in range(0,sample_size):
                    ax[j,i].plot(bid_result[bids[i,j]][p1,p2,:])
        elif "DA" in bids[i,j]:
            for p1 in range(0,sample_size):
                ax[j,i].plot(bid_result[bids[i,j]][p1,:])
        else:
            ax[j,i].plot(bid_result[bids[i,j]])
        
        ax[j,i].set_title(bids[i,j])


prices = np.array([['f_FD2_up_tw_input','f_FD2_dn_tw_input'],['f_DA_tw_input','f_DA_tw_input'],['f_FD1_up_tw_input','f_FD1_dn_tw_input']])

fig, ax = plt.subplots(2,3)
print(bid_result['f_FD2_up_tw_input'])
print(bid_result['f_FD2_y_up_tw_input'])
print(bid_result['f_FD2_up_tw_input']*bid_result['f_FD2_y_up_tw_input'])
f_FD2_up_tw_input=  bid_result['f_FD2_up_tw_input']*bid_result['f_FD2_y_up_tw_input']
f_FD2_dn_tw_input = bid_result['f_FD2_dn_tw_input']*bid_result['f_FD2_y_dn_tw_input']
f_DA_tw_input = bid_result['f_DA_tw_input']

f_FD1_up_tw_input = bid_result['f_FD1_up_tw_input']*bid_result['f_FD1_y_up_tw_input']
f_FD1_dn_tw_input = bid_result['f_FD1_dn_tw_input']*bid_result['f_FD1_y_dn_tw_input']


print(np.shape(f_FD2_up_tw_input))
for six in range(0,sample_size):
    ax[0,0].plot(f_FD2_up_tw_input[six,:])
    ax[0,0].set_title('f_FD2_up_tw_input')
    ax[1,0].plot(f_FD2_dn_tw_input[six,:])
    ax[1,0].set_title('f_FD2_dn_tw_input')

    print(np.shape(f_DA_tw_input))
    ax[0,1].plot(f_DA_tw_input[six,0,:])
    ax[0,1].set_title('f_DA_tw_input')

    ax[0,2].plot(f_FD1_up_tw_input[six,0,0,:])
    ax[0,2].set_title('f_FD1_up_tw_input')
    ax[1,2].plot(f_FD1_dn_tw_input[six,0,0,:])
    ax[1,2].set_title('f_FD1_dn_tw_input')


x = [i for i in range(0,24)]
print(x)
print(np.size(x),np.size(b_FD2_up_all))
print(b_FD2_up_all)
plt.show()


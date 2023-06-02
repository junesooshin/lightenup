
## Helper functions for Model comparison and analysis
 
# importing the module
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def json_to_df(json_file_path, orient='columns'):
    """
    Function that loads a JSON file and converts it to a Pandas DataFrame.
    Raises an exception if keys have different lengths.

    Parameters:
    json_file_path (str): Path to the JSON file.
    orient (str): Orientation of the DataFrame. Valid values are 'columns' (default) and 'index'.

    Returns:
    Pandas DataFrame
    """
    
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except IOError:
        print(f"Error: Failed to read file {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON data in file {json_file_path}")
        return None
    
    # Check if keys have equal lengths
    lengths = [len(value) for value in json_data.values()]
    if len(set(lengths)) != 1:
        raise ValueError("JSON file has keys with different lengths")
    
    data_dict = {key: value for key, value in json_data.items()} # Transform to dictionary
    df = pd.DataFrame.from_dict(data_dict, orient=orient) # Transform to dataframe. 
    return df


# Plotting function
def plot_Actions(SOC, Spot_action, FCR_D_Up_action, FCR_D_Dn_action, Spot_price, FCR_D_Up_price, FCR_D_Dn_price, Threshold_Max=0, Threshold_Min=0):
    
    # Input
    # SOC            , State of charge [24,]. Values are either 0 or 1       
    # Spot_action    , The actions in spot [24,]
    # FCR_D_Up_action, The actions in upregulation FCR-D [24,]
    # FCR_D_Dn_action, The actions in downregulation FCR-D [24,]
    # Spot_price     , The spot price [24,]
    # FCR_D_Up_price , The FCR-D up price [24,]
    # FCR_D_Up_price , The FCR-D down price [24,]
    # Threshold_Max  , Upperbound for when spot activates to make the vertical axis
    # Threshold_Min  , lowerbound for when spot activates
    
    
    fig1, ax0 = plt.subplots()

    ax0.plot(SOC, color='C0', label='SOC')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('SOC')
    ax0.tick_params(axis='y', labelcolor='C0')

    ax1 = ax0.twinx()
    ax1.plot(np.arange(24),FCR_D_Up_price, 'o-', color='C1', label='FCR-D Up Price')
    ax1.plot(np.arange(24),FCR_D_Dn_price, 'o-', color='C2', label='FCR-D Dn Price')

    # create a single legend for all the plots
    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1
    fig1.legend(handles, labels, loc='center')

    fig2, ax2 = plt.subplots()
    # create a second y-axis for the price data
    ax3 = ax2.twinx()

    # plot the Spot price and FCR_D_Up price on the second y-axis
    ax2.plot(np.arange(24),Spot_price, 'o-', color='C1', label='Spot Price')
    ax2.set_ylabel('Price [EUR/MW]')
    ax2.tick_params(axis='y', labelcolor='C1')
    
    if Threshold_Max != 0 and Threshold_Min != 0:
        ax2.axhline(Threshold_Max, linestyle='--', color='C1')
        ax2.axhline(Threshold_Min, linestyle='--', color='C1')
    

    mask_spot = (Spot_action != 0) 
    mask_FCRD_Up = (FCR_D_Up_action != 0)
    mask_FCRD_Dn = (FCR_D_Dn_action != 0)
    x_spot = np.arange(24)[mask_spot]
    y_spot = Spot_action[mask_spot]
    x_FCRD_Up = np.arange(24)[mask_FCRD_Up]
    y_FCRD_Up = FCR_D_Up_action[mask_FCRD_Up]
    x_FCRD_Dn = np.arange(24)[mask_FCRD_Dn]
    y_FCRD_Dn = FCR_D_Dn_action[mask_FCRD_Dn]

    # plot the non-zero action values on the first y-axis
    ax3.scatter(x_spot, y_spot, color='C3', label='Spot Action')
    ax3.scatter(x_FCRD_Up, y_FCRD_Up, color='C4', label='FCR_D_Up Action')
    ax3.scatter(x_FCRD_Dn, y_FCRD_Dn, color='C5', label='FCR_D_Dn Action')
    ax3.set_ylabel('Action')
    ax3.tick_params(axis='y', labelcolor='C3')

    # create a single legend for all the plots
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles = handles2 + handles3
    labels = labels2 + labels3
    fig2.legend(handles, labels, loc='center')


    
    return plt.show()

def plot_Accepted_Bids(df_input_FD2,df_input_DA,df_input_FD1,df_input_RT,figname):

    '''

    Input:  Dataframe with the following columns:
    "p_DA_up" , "p_DA_dn" , "p_FD1_up" , "p_FD1_dn" , "p_FD2_up" , "p_FD2_dn" , "p_Bal_up" , "p_Bal_dn" , "SOC"

    '''

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i, df_input in enumerate([df_input_FD2,df_input_DA,df_input_FD1,df_input_RT]):
        # Extract data from the dataframe
        #print(df_input)
        x = df_input.index
        y1 = df_input["p_DA_up"]
        y2 = -df_input["p_DA_dn"]
        y3 = df_input["p_FD1_up"]
        y4 = -df_input["p_FD1_dn"]
        y5 = df_input["p_FD2_up"]
        y6 = -df_input["p_FD2_dn"]
        y7 = df_input["p_Bal_up"]
        y8 = -df_input["p_Bal_dn"]
        y9 = df_input["SOC"]
    
        # Create the figure and axis objects
        ax2 = axs[i // 2, i % 2].twinx()
        
        # Plot the data on the first y-axis
        axs[i // 2, i % 2].bar(x, y1, label="p_DA_up", color="C0")
        axs[i // 2, i % 2].bar(x, y2, bottom=0, label="p_DA_dn", color="C1")
        axs[i // 2, i % 2].bar(x, y3, bottom=y1, label="p_FD1_up", color="C2")
        axs[i // 2, i % 2].bar(x, y4, bottom=y2, label="p_FD1_dn", color="C3")
        axs[i // 2, i % 2].bar(x, y5, bottom=y1+y3, label="p_FD2_up", color="C4")
        axs[i // 2, i % 2].bar(x, y6, bottom=y2+y4, label="p_FD2_dn", color="C5")
        axs[i // 2, i % 2].bar(x, y7, bottom=y1+y3+y5, label="p_Bal_up", color="C6")
        axs[i // 2, i % 2].bar(x, y8, bottom=y2+y4+y6, label="p_Bal_dn", color="C7")
        
        # Plot the data on the secondary y-axis
        ax2.plot(x, y9, label="SOC", color="C9", linewidth=2)

        # create a legend and set its location
        lines, labels = axs[i // 2, i % 2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # save the figure with the specified file name and path

    # set the x-axis label and title
    axs[1, 0].set_xlabel('Hour')
    axs[1, 1].set_xlabel('Hour')

    axs[0, 0].set_title('Accepted Bids')
    axs[0, 1].set_title('Accepted Bids')

    # set the y-axis labels
    axs[0,0].set_ylabel('Power (MW)')
    axs[1,0].set_ylabel('Power (MW)')
    ax2.set_ylabel('SOC (%)')
    plt.show()
    #fig.savefig(figname)
  
def plot_Accepted_Bids2(df_input_FD2,df_input_DA,df_input_FD1,df_input_RT,figname):

    '''

    Input:  Dataframe with the following columns:
    "p_DA_up" , "p_DA_dn" , "p_FD1_up" , "p_FD1_dn" , "p_FD2_up" , "p_FD2_dn" , "p_Bal_up" , "p_Bal_dn" , "SOC"

    '''

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    handles, labels = [], []

    for i, df_input in enumerate([df_input_FD2,df_input_DA,df_input_FD1,df_input_RT]):
        # Extract data from the dataframe
        #print(df_input)
        x = df_input.index
        y1 = df_input["p_DA_up"]
        y2 = -df_input["p_DA_dn"]
        y3 = df_input["p_FD1_up"]
        y4 = -df_input["p_FD1_dn"]
        y5 = df_input["p_FD2_up"]
        y6 = -df_input["p_FD2_dn"]
        y9 = df_input["SOC"]
    
        # Create the figure and axis objects
        ax2 = axs[i // 2, i % 2].twinx()
        
        # Plot the data on the first y-axis
        axs[i // 2, i % 2].bar(x, y1, label="p_DA_up", color="C0")
        axs[i // 2, i % 2].bar(x, y2, bottom=0, label="p_DA_dn", color="C1")
        axs[i // 2, i % 2].bar(x, y3, bottom=y1, label="p_FD1_up", color="C2")
        axs[i // 2, i % 2].bar(x, y4, bottom=y2, label="p_FD1_dn", color="C3")
        axs[i // 2, i % 2].bar(x, y5, bottom=y1+y3, label="p_FD2_up", color="C4")
        axs[i // 2, i % 2].bar(x, y6, bottom=y2+y4, label="p_FD2_dn", color="C5")
        
        # Plot the data on the secondary y-axis
        ax2.plot(x, y9, label="SOC", color="C9", linewidth=2)

        # get the handles and labels of the lines in the subplot
        h, l = axs[0, 0].get_legend_handles_labels()
        handles += h
        labels += l
        
    # create the legend outside the subplots
    fig.legend(handles, labels, loc='upper center', ncol=5)
    plt.show()



def Create_Array_from_Rev(df_Exp_rev,df_RT_rev):
    # Initialize a numpy array of size (month, days, models, result) with empty values
    Array = np.empty((12, 88, 4, 2))

    for col in df_Exp_rev.columns:
        
        # Create a four dimensional 

        # (M, D, Model)
        
        # Assign the month, the day, the model, and each of the 

        # Extract the 'm', 'd', and 't' values from column name
        parts = col.split('_')
        m = int(parts[0][1:])
        d = int(parts[1][1:])
        t = int(parts[2][1:])



        Array[(m-1),(d-1),0,0] = df_Exp_rev[col]['rule']
        Array[(m-1),(d-1),1,0] = df_Exp_rev[col]['det']
        Array[(m-1),(d-1),2,0] = df_Exp_rev[col]['sto']
        Array[(m-1),(d-1),3,0] = df_Exp_rev[col]['learn']

        Array[(m-1),(d-1),0,1] = df_RT_rev[col]['rule']
        Array[(m-1),(d-1),1,1] = df_RT_rev[col]['det']
        Array[(m-1),(d-1),2,1] = df_RT_rev[col]['sto']
        Array[(m-1),(d-1),3,1] = df_RT_rev[col]['learn']
        
    return Array


def plot_Revenue_Test(Array):
    # Assuming Array[m,d,model,rev] is the four-dimensional array

    # Set up the plot
    #fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    # Define a list of colors for models
    colors = ['red' , 'blue', 'green', 'orange']
    Model  = ['Rule', 'Det' , 'Sto'  , 'Learn' ]

    # Define a list of symbols for revs
    symbols = ['o'  , 's' ]
    Rev     = ['Exp', 'RT']

    # Define a list of x-axis movement for each rev
    xaxis_move = [-0.2,-0.15,-0.1,-0.05, 0.05,0.1,0.15,0.2]

    marker_size = 2
    marker_size_2 = 20

    # Calculate the mean along the second axis (axis=1)
    mean_array_along_d = np.mean(Array, axis=1, keepdims=True)

    # Iterate over d values
    for d in range(Array.shape[1]):
        # Iterate over rev values
        count = 0
        for rev in range(2):
            # Iterate over model values
            for model in range(4):
                
                # Extract the data for the current d, model, and rev
                data = Array[:, d, model, rev]

                # Get the unique color and symbol based on the model and rev
                color = colors[model]
                symbol = symbols[rev]

                # Calculate the modified x values based on the rev
                
                modified_x = list(range(Array.shape[0]))  # Convert range object to a list
                modified_x = [x + xaxis_move[count] for x in modified_x]
                count = count + 1 # For modified_x

                # Plot the data with the unique color and symbol
                ax1.scatter(modified_x, data, color=color, marker=symbol,s=marker_size)


    # Set y-axis label
    ax1.set_ylabel('Revenue [EUR]')

    # Create a legend for colors and symbols
    legend_handles = []
    for model, color in enumerate(colors):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f'{Model[model]}', markerfacecolor=color, markersize=10))
    for rev, symbol in enumerate(symbols):
        legend_handles.append(Line2D([0], [0], marker=symbol, color='w', label=f'{Rev[rev]}', markerfacecolor='black', markersize=10))

    # Add the legend to the plot outside the graph area
    ax1.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 0.7))

    # The other subplot
    count = 0
    for rev in range(2):
        # Iterate over model values
        for model in range(4):

            # Extract the data for the current d, model, and rev
            data = mean_array_along_d[:, 0, model, rev]

            # Get the unique color and symbol based on the model and rev
            color = colors[model]
            symbol = symbols[rev]

            # Calculate the modified x values based on the rev
            modified_x = list(range(Array.shape[0]))  # Convert range object to a list
            modified_x = [x + xaxis_move[count] for x in modified_x]
            count = count + 1  # For modified_x

            # Plot the data with the unique color and symbol
            ax2.scatter(modified_x, data, color=color, marker=symbol, s=marker_size_2)

    # Set y-axis label
    ax2.set_ylabel('Revenue [EUR]')

    # Set x-axis label
    ax2.set_xlabel('Month of training')
    ax2.set_xticks(range(0, Array.shape[0]),[1,2,3,4,5,6,7,8,9,10,11,12]) 

    # Set title for ax1
    ax1.set_title('Revenue for different training sizes and at different test days')

    # Set title for ax2
    ax2.set_title('Mean revenue for different training sizes')

    # Show the plot
    plt.show()


def import_test_case(current_directory, choose_id):
    #Import JSON files
    with open(current_directory + f'/Results/det_{choose_id}.json') as results_det_json:
        Results_det = json.load(results_det_json)
    with open(current_directory + f'/Results/learn_{choose_id}.json') as results_learn_json:
        Results_learn = json.load(results_learn_json)
    with open(current_directory + f'/Results/sto_{choose_id}.json') as results_sto_json:
        Results_sto = json.load(results_sto_json)
    with open(current_directory + f'/Results/rule_{choose_id}.json') as results_rule_json:
        Results_rule = json.load(results_rule_json)

    #Convert results in to np.arrays
    det_bid = {}
    det_RT = {}
    learn_bid = {}
    learn_RT = {}
    sto_bid = {}
    sto_RT = {}
    rule_bid = {}
    rule_RT = {}

    for key, value in Results_det['Bid'].items():
        det_bid[key] = np.array(value)
    for key, value in Results_det['RT'].items():
        det_RT[key] = np.array(value)

    for key, value in Results_learn['Bid'].items():
        learn_bid[key] = np.array(value)
    for key, value in Results_learn['RT'].items():
        learn_RT[key] = np.array(value)

    for key, value in Results_sto['Bid'].items():
        sto_bid[key] = np.array(value)
    for key, value in Results_sto['RT'].items():
        sto_RT[key] = np.array(value)

    for key, value in Results_rule['Bid'].items():
        rule_bid[key] = np.array(value)
    for key, value in Results_rule['RT'].items():
        rule_RT[key] = np.array(value)

    return det_bid, det_RT, learn_bid, learn_RT, sto_bid, sto_RT, rule_bid, rule_RT


def bid_plots(bid_result, model, color):
    #Bid quantity plots with SOC
    fig, ax1 = plt.subplots(figsize=(7,5))

    x = np.arange(1,25)

    mask = np.nonzero(bid_result['b_FD1_up'])[0]
    ax1.scatter(x[mask], bid_result['b_FD1_up'][mask], label='b_FD1_up', marker='x',s=50, color=color['FD1_up'])
    mask = np.nonzero(bid_result['b_FD2_up'])[0]
    ax1.scatter(x[mask], bid_result['b_FD2_up'][mask], label='b_FD2_up', marker='x',s=50, color=color['FD2_up'])
    mask = np.nonzero(bid_result['b_FD1_dn'])[0]
    ax1.scatter(x[mask], bid_result['b_FD1_dn'][mask], label='b_FD1_dn', marker='o',s=50, color=color['FD1_dn'])
    mask = np.nonzero(bid_result['b_FD2_dn'])[0]
    ax1.scatter(x[mask], bid_result['b_FD2_dn'][mask], label='b_FD2_dn', marker='o',s=50, color=color['FD2_dn'])
    mask = np.nonzero(bid_result['b_DA_dn'])[0]
    ax1.scatter(x[mask], bid_result['b_DA_dn'][mask], label='b_DA_dn', marker='1',s=50, color=color['DA_dn'])
    mask = np.nonzero(bid_result['b_DA_up'])[0]
    ax1.scatter(x[mask], bid_result['b_DA_up'][mask], label='b_DA_up', marker='1',s=50, color=color['DA_up'])

    ax1.set_ylabel('Bid quantities [MW]', fontsize=12)
    ax1.set_xlabel('Hours', fontsize=12)
    ax1.set_xticks([1, 6, 12, 18, 24])
    # ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=3)
    ax1.set_title(model, fontsize=14) # Remove this when exporting plots for overleaf

    ax2_1 = ax1.twinx()
    if 'Rule' not in model:
        ax2_1.plot(x, bid_result['SOC']/6, color=color['SOC'], label='SOC')
    elif 'Rule' in model: #Rule based model gives normalized SOC
        ax2_1.plot(x, bid_result['SOC'], color=color['SOC'], label='SOC')
    ax2_1.set_ylabel('SOC [p.u.]', fontsize=12)
    ax2_1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2_1.set_ylim(-0.05,1.05)
    # ax2_1.legend(loc='upper right', bbox_to_anchor=(1, -0.1), ncol=3)

    # Combine the legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2_1.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, -0.15), ncol=4)

    plt.show()

def plot_training_price(with_acceptance, bid_result, model, color):
    # Plot prices used for training
    fig, ax2 = plt.subplots(figsize=(7,5))

    x = np.arange(1,25)
    if 'Det' in model or 'Rule' in model:
        if with_acceptance == True:
            ax2.plot(x, bid_result['f_lambda_FD2_up']*bid_result['f_FD2_y_up_t'], label='f_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, bid_result['f_lambda_FD2_dn']*bid_result['f_FD2_y_dn_t'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
            if 'Det' in model:
                ax2.plot(x, bid_result['f_lambda_FD1_up']*bid_result['f_FD1_y_up_t'], label='f_FD1_up', marker='.', color=color['FD1_up'])
                ax2.plot(x, bid_result['f_lambda_FD1_dn']*bid_result['f_FD1_y_dn_t'], label='f_FD1_dn', marker='.', color=color['FD1_dn'])
        elif with_acceptance == False:
            ax2.plot(x, bid_result['f_lambda_FD2_up'], label='f_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, bid_result['f_lambda_FD2_dn'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
            if 'Det' in model:
                ax2.plot(x, bid_result['f_lambda_FD1_up'], label='f_FD1_up', marker='.', color=color['FD1_up'])
                ax2.plot(x, bid_result['f_lambda_FD1_dn'], label='f_FD1_dn', marker='.', color=color['FD1_dn'])
        ax2.plot(x, bid_result['f_DA_t'], label='f_DA', marker='.', color=color['DA_up']) #Expected DA price
        if 'Rule' in model:
            ax2.plot(x, bid_result['Threshold_Max'], linestyle='--', color='grey')
            ax2.plot(x, bid_result['Threshold_Min'], linestyle='--', color='grey')

    elif 'Learn' in model: # This only plots training data for now, the bid price is equal to the forecast price
        X_train_spot = np.array(bid_result["X"])[0,:,:].T
        X_train_FD1_dn = np.array(bid_result["X"])[1,:,:].T
        X_train_FD2_dn = np.array(bid_result["X"])[2,:,:].T
        X_train_FD1_up = np.array(bid_result["X"])[3,:,:].T
        X_train_FD2_up = np.array(bid_result["X"])[4,:,:].T

        ax2.plot(x, X_train_FD2_up.mean(axis=1), label='X_FD2_up', marker='.', color=color['FD2_up'])
        ax2.plot(x, X_train_FD2_dn.mean(axis=1), label='X_FD2_dn', marker='.', color=color['FD2_dn'])
        ax2.plot(x, X_train_FD1_up.mean(axis=1), label='X_FD1_up', marker='.', color=color['FD1_up'])
        ax2.plot(x, X_train_FD1_dn.mean(axis=1), label='X_FD1_dn', marker='.', color=color['FD1_dn'])
        ax2.plot(x, X_train_spot.mean(axis=1), label='X_spot', marker='.', color=color['DA_up'])

    elif 'Sto' in model:
        FD2_up_mean = np.mean(bid_result['f_FD2_up_tw_input'], axis=0)
        FD2_up_max = np.max(bid_result['f_FD2_up_tw_input'], axis=0)
        FD2_up_min = np.min(bid_result['f_FD2_up_tw_input'], axis=0)

        FD2_dn_mean = np.mean(bid_result['f_FD2_dn_tw_input'], axis=0)
        FD2_dn_max = np.max(bid_result['f_FD2_dn_tw_input'], axis=0)
        FD2_dn_min = np.min(bid_result['f_FD2_dn_tw_input'], axis=0)

        DA_mean = np.mean(np.reshape(bid_result['f_DA_tw_input'], (3*2,24)), axis=0)
        DA_max = np.max(np.reshape(bid_result['f_DA_tw_input'], (3*2,24)), axis=0)
        DA_min = np.min(np.reshape(bid_result['f_DA_tw_input'], (3*2,24)), axis=0)

        FD1_up_mean = np.mean(np.reshape(bid_result['f_FD1_up_tw_input'], (5*3*2,24)), axis=0)
        FD1_up_max = np.max(np.reshape(bid_result['f_FD1_up_tw_input'], (5*3*2,24)), axis=0)
        FD1_up_min = np.min(np.reshape(bid_result['f_FD1_up_tw_input'], (5*3*2,24)), axis=0)

        FD1_dn_mean = np.mean(np.reshape(bid_result['f_FD1_dn_tw_input'], (5*3*2,24)), axis=0)
        FD1_dn_max = np.max(np.reshape(bid_result['f_FD1_dn_tw_input'], (5*3*2,24)), axis=0)
        FD1_dn_min = np.min(np.reshape(bid_result['f_FD1_dn_tw_input'], (5*3*2,24)), axis=0)

        FD2_up_accept_mean = np.mean(bid_result['f_FD2_y_up_tw_input'], axis=0)
        FD2_up_accept_max = np.max(bid_result['f_FD2_y_up_tw_input'], axis=0)
        FD2_up_accept_min = np.min(bid_result['f_FD2_y_up_tw_input'], axis=0)

        FD2_dn_accept_mean = np.mean(bid_result['f_FD2_y_dn_tw_input'], axis=0)
        FD2_dn_accept_max = np.max(bid_result['f_FD2_y_dn_tw_input'], axis=0)
        FD2_dn_accept_min = np.min(bid_result['f_FD2_y_dn_tw_input'], axis=0)

        FD1_up_accept_mean = np.mean(np.reshape(bid_result['f_FD1_y_up_tw_input'], (5*3*2,24)), axis=0)
        FD1_up_accept_max = np.max(np.reshape(bid_result['f_FD1_y_up_tw_input'], (5*3*2,24)), axis=0)
        FD1_up_accept_min = np.min(np.reshape(bid_result['f_FD1_y_up_tw_input'], (5*3*2,24)), axis=0)

        FD1_dn_accept_mean = np.mean(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (5*3*2,24)), axis=0)
        FD1_dn_accept_max = np.max(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (5*3*2,24)), axis=0)
        FD1_dn_accept_min = np.min(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (5*3*2,24)), axis=0)

        if with_acceptance == True:
            ax2.plot(x, FD2_dn_mean*FD2_dn_accept_mean, label='Mean D-2 dn',color=color['FD2_dn'], marker='.')
            ax2.fill_between(x, FD2_dn_min*FD2_dn_accept_min, FD2_dn_max*FD2_dn_accept_max, alpha=0.2,color=color['FD2_dn'])
            ax2.plot(x, FD2_up_mean*FD2_up_accept_mean, label='Mean D-2 up',color=color['FD2_up'], marker='.')
            ax2.fill_between(x, FD2_up_min*FD2_up_accept_min, FD2_up_max*FD2_up_accept_max, alpha=0.2,color=color['FD2_up'])
            ax2.plot(x, FD1_dn_mean*FD1_dn_accept_mean, label='Mean D-1 dn',color=color['FD1_dn'], marker='.')
            ax2.fill_between(x, FD1_dn_min*FD1_dn_accept_min, FD1_dn_max*FD1_dn_accept_max, alpha=0.2,color=color['FD1_dn'])
            ax2.plot(x, FD1_up_mean*FD1_up_accept_mean, label='Mean D-1 up',color=color['FD1_up'], marker='.')
            ax2.fill_between(x, FD1_up_min*FD1_up_accept_min, FD1_up_max*FD1_up_accept_max, alpha=0.2,color=color['FD1_up'])
            ax2.plot(x, DA_mean, label='Mean DA',color=color['DA_up'], marker='.')
            ax2.fill_between(x, DA_min, DA_max, alpha=0.2,color=color['DA_up'])
        elif with_acceptance == False:    
            ax2.plot(x, FD2_dn_mean, label='Mean D-2 dn',color=color['FD2_dn'], marker='.')
            ax2.fill_between(x, FD2_dn_min, FD2_dn_max, alpha=0.2,color=color['FD2_dn'])
            ax2.plot(x, FD2_up_mean, label='Mean D-2 up',color=color['FD2_up'], marker='.')
            ax2.fill_between(x, FD2_up_min, FD2_up_max, alpha=0.2,color=color['FD2_up'])
            ax2.plot(x, FD1_dn_mean, label='Mean D-1 dn',color=color['FD1_dn'], marker='.')
            ax2.fill_between(x, FD1_dn_min, FD1_dn_max, alpha=0.2,color=color['FD1_dn'])
            ax2.plot(x, FD1_up_mean, label='Mean D-1 up',color=color['FD1_up'], marker='.')
            ax2.fill_between(x, FD1_up_min, FD1_up_max, alpha=0.2,color=color['FD1_up'])
            ax2.plot(x, DA_mean, label='Mean DA',color=color['DA_up'], marker='.')
            ax2.fill_between(x, DA_min, DA_max, alpha=0.2,color=color['DA_up'])

    if with_acceptance == True:
        ax2.set_ylabel('Price used for training with acceptance [EUR/MW]', fontsize=12)
    elif with_acceptance == False:
        ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)
    ax2.set_xlabel('Hours', fontsize=12)
    ax2.set_xticks([1, 6, 12, 18, 24])
    ax2.legend(loc='upper left', bbox_to_anchor=(0.15, -0.15), ncol=3)
    ax2.set_title(model, fontsize=14)

    plt.show()

def plot_bidding_price(bid_result, model, color):
    # Plot FCR-D bidding prices
    fig, ax3 = plt.subplots(figsize=(7,5))
    x = np.arange(1,25)

    ax3.plot(x, bid_result['f_lambda_FD2_up'], label='FD2_up', marker='.', color=color['FD2_up'])
    ax3.plot(x, bid_result['f_lambda_FD2_dn'], label='FD2_dn', marker='.', color=color['FD2_dn'])
    if 'Rule' not in model:
        ax3.plot(x, bid_result['f_lambda_FD1_up'], label='FD1_up', marker='.', color=color['FD1_up'])
        ax3.plot(x, bid_result['f_lambda_FD1_dn'], label='FD1_dn', marker='.', color=color['FD1_dn'])

    ax3.set_ylabel('Bid Price [EUR/MW]', fontsize=12)
    ax3.set_xlabel('Hours', fontsize=12)
    ax3.set_xticks([1, 6, 12, 18, 24])
    ax3.legend(loc='upper left', bbox_to_anchor=(0.15, -0.15), ncol=3)
    ax3.set_title(model, fontsize=14)

    plt.show()

def plot_accepted_price(data_RT_input, model):
    # Plot RT accepted prices and RT DA price
    fig, ax = plt.subplots(2,2, figsize=(16, 10))
    x = np.arange(1,25)

    ax[0,0].plot(x, data_RT_input['FD1_up_bid_price'], label='Bid price', marker = '.')
    ax[0,0].plot(x, data_RT_input['FD1_up_price_test'], label='Real market price', marker = '.')
    ax[0,0].set_xlabel('Hours', fontsize=12)
    ax[0,0].set_ylabel('Price [EUR]', fontsize=12)
    ax[0,0].set_title('FCR D-1 up price acceptance in RT')
    ax[0,0].set_xticks([1,6,12,18,24])

    ax1_0 = ax[0,0].twinx()
    ax1_0.scatter(x, data_RT_input['acceptance_FD1_up'], color='black', label='acceptance', marker = '.')
    ax1_0.set_ylabel('acceptance', fontsize=12)
    ax1_0.set_ylim(-0.05,1.05)
    ax1_0.set_yticks([0,1])

    ax[0,1].plot(x, data_RT_input['FD2_up_bid_price'], label='Bid price', marker = '.')
    ax[0,1].plot(x, data_RT_input['FD2_up_price_test'], label='Real market price', marker = '.')
    #ax[0,1].legend(loc='center left', bbox_to_anchor=(0.2,-0.2), ncol=2)
    ax[0,1].set_xlabel('Hours', fontsize=12)
    ax[0,1].set_ylabel('Price [EUR]', fontsize=12)
    ax[0,1].set_title('FCR D-2 up price acceptance in RT')
    ax[0,1].set_xticks([1,6,12,18,24])

    ax1_1 = ax[0,1].twinx()
    ax1_1.scatter(x, data_RT_input['acceptance_FD2_up'], color='black', label='acceptance', marker = '.')
    ax1_1.set_ylabel('acceptance', fontsize=12)
    ax1_1.set_ylim(-0.05,1.05)
    ax1_1.set_yticks([0,1])

    ax[1,0].plot(x, data_RT_input['FD1_dn_bid_price'], label='Bid price', marker = '.')
    ax[1,0].plot(x, data_RT_input['FD1_down_price_test'], label='Real market price', marker = '.')
    ax[1,0].legend(loc='center left', bbox_to_anchor=(0.2,-0.2), ncol=2)
    ax[1,0].set_xlabel('Hours', fontsize=12)
    ax[1,0].set_ylabel('Price [EUR]', fontsize=12)
    ax[1,0].set_title('FCR D-1 down price acceptance in RT')
    ax[1,0].set_xticks([1,6,12,18,24])

    ax1_2 = ax[1,0].twinx()
    ax1_2.scatter(x, data_RT_input['acceptance_FD1_down'], color='black', label='acceptance', marker = '.')
    ax1_2.set_ylabel('acceptance', fontsize=12)
    ax1_2.set_ylim(-0.05,1.05)
    ax1_2.set_yticks([0,1])

    ax[1,1].plot(x, data_RT_input['FD2_dn_bid_price'], label='Bid price', marker = '.')
    ax[1,1].plot(x, data_RT_input['FD2_down_price_test'], label='Real market price', marker = '.')
    ax[1,1].legend(loc='center left', bbox_to_anchor=(0.2,-0.2), ncol=2)
    ax[1,1].set_xlabel('Hours', fontsize=12)
    ax[1,1].set_ylabel('Price [EUR]', fontsize=12)
    ax[1,1].set_title('FCR D-2 down price acceptance in RT')
    ax[1,1].set_xticks([1,6,12,18,24])

    ax1_3 = ax[1,1].twinx()
    ax1_3.scatter(x, data_RT_input['acceptance_FD2_down'], color='black', label='acceptance', marker = '.')
    ax1_3.set_ylabel('acceptance', fontsize=12)
    ax1_3.set_ylim(-0.05,1.05)
    ax1_3.set_yticks([0,1])

    fig.suptitle(model, fontsize=12) #Remove this for overleaf export

def plot_coefficients(learn_bid):
    # Plot coefficients
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up", "1"]
    q_FD2_up = learn_bid["q_FD2_up"]
    q_FD2_dn = learn_bid["q_FD2_dn"]
    q_FD1_up = learn_bid["q_FD1_up"]
    q_FD1_dn = learn_bid["q_FD1_dn"]
    q_DA_up = learn_bid["q_DA_up"]
    q_DA_dn = learn_bid["q_DA_dn"]

    fig, ax4 = plt.subplots(figsize=(7,5))

    ax4.set_ylabel('Coefficients', fontsize=12)
    ax4.set_xlabel('Features', fontsize=12)
    ax4.plot(q_FD2_up, label='q_FD2_up', marker = '.')
    ax4.plot(q_FD2_dn, label='q_FD2_dn', marker = '.')
    ax4.plot(q_FD1_up, label='q_FD1_up', marker = '.')
    ax4.plot(q_FD1_dn, label='q_FD1_dn', marker = '.')
    ax4.plot(q_DA_up, label='q_DA_up', marker = '.')
    ax4.plot(q_DA_dn, label='q_DA_dn', marker = '.')
    ax4.set_xticks([i for i in range(0,len(q_FD2_up))], Feature_Selection, rotation=40)
    ax4.legend(loc='upper left', bbox_to_anchor=(0.2, -0.25), ncol=2, fontsize=12)

    plt.show()

def plot_exp_and_RT_revenue(det_RT, det_bid, sto_RT, sto_bid, learn_RT, learn_bid, rule_RT, rule_bid):
    # Plot RT and expected revenue
    fig, ax6 = plt.subplots(figsize=(7,5))
    x = np.arange(1,25)

    ax6.plot(x, det_RT['revenue_t'], label='Det_RT', marker = '.', color='C0') 
    ax6.plot(x, det_bid['obj_t'], label='Det_Exp', linestyle='dashed', color='C0') 

    ax6.plot(x, sto_RT['revenue_t'], label='Sto_RT', marker = '.', color='C1') 
    ax6.plot(x, sto_bid['obj_t'], label='Sto_Exp', linestyle='dashed', color='C1') 

    ax6.plot(x, learn_RT['revenue_t'], label='Learn_RT', marker = '.', color='C2') 
    ax6.plot(x, learn_bid['obj_t'].flatten(), label='Learn_Exp', linestyle='dashed', color='C2') 

    ax6.plot(x, rule_RT['revenue_t'], label='Rule_RT', marker = '.', color='C3') 
    ax6.plot(x, rule_bid['obj_t'], label='Rule_Exp', linestyle='dashed', color='C3') 

    ax6.set_ylabel('Revenue [EUR]', fontsize=12)
    ax6.set_xlabel('Hours', fontsize=12)
    ax6.set_xticks([1,6,12,18,24])
    ax6.legend(loc='upper left', bbox_to_anchor=(-0.05, -0.15), ncol=4, fontsize=12)

    plt.show()

def plot_battery_dynamics(RTresult):
    # Plot battery dynamics
    fig, ax5 = plt.subplots(figsize=(7, 5))
    x = np.arange(1,25)

    ax5.plot(x, RTresult['p_all_up'], label='p_all_up', marker = '.')
    ax5.plot(x, RTresult['p_all_dn'], label='p_all_dn', marker = '.')
    ax5.plot(x, RTresult['failure_up'], label='failure_up')
    ax5.plot(x, RTresult['failure_dn'], label='failure_dn')
    ax5.plot(x, RTresult['p_dis_max']*np.ones(24), '--' ,label='p_dis_max', color='grey')
    ax5.plot(x, RTresult['p_ch_max']*np.ones(24), '--' ,label='p_ch_max', color='grey')
    ax5.legend(loc='center left', bbox_to_anchor=(0.15,-0.2), ncol=3)
    ax5.set_xlabel('Hours')
    ax5.set_ylabel('Power [MW]')
    ax5.set_xticks([1,6,12,18,24])

    ax5_1 = ax5.twinx()
    ax5_1.plot(x, RTresult['SOC']/6, color='black', label='SOC', marker = '.')
    ax5_1.set_ylabel('SOC [p.u.]', fontsize=12)
    ax5_1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax5_1.set_ylim([-0.05,1.05])

    plt.show()
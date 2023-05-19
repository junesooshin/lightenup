
###############################################
########### PLOTTING ##############
############################## 
 
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

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





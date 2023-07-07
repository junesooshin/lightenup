
## Helper functions for Model comparison and analysis
 
# importing the module
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from PIL import Image
from matplotlib.ticker import ScalarFormatter

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


def Create_Array_from_Profit(df_Exp_profit,df_RT_profit,SampleSizes, NumForecasts,):
    # Initialize a numpy array of size (number of different forecast accuracies,number of different train sizes, test days, models, result) with empty values
    Array = np.empty((NumForecasts, len(SampleSizes), 88, 5, 2, 1))

    for col in df_Exp_profit.columns:
        
        # Create a four dimensional 

        # (M, D, Model)
        
        # Assign the month, the day, the model, and each of the 

        # Extract the 'm', 'd', and 't' values from column name
        parts = col.split('_')

        Samplesize = int(parts[1][1:])
        
        # Only if the file is part of the specified range then include it.
        if Samplesize in SampleSizes: 
            m = SampleSizes.index(Samplesize)
        else:
            continue
        
        f = int(parts[0][1:])
        
        d = int(parts[2][3:])
        t = int(parts[3][1:])



        Array[(f-1),(m),(d-1),0,0,0] = df_Exp_profit[col]['rule']
        Array[(f-1),(m),(d-1),1,0,0] = df_Exp_profit[col]['det']
        Array[(f-1),(m),(d-1),2,0,0] = df_Exp_profit[col]['sto']
        Array[(f-1),(m),(d-1),3,0,0] = df_Exp_profit[col]['feature']
        Array[(f-1),(m),(d-1),4,0,0] = df_Exp_profit[col]['oracle']

        Array[(f-1),(m),(d-1),0,1,0] = df_RT_profit[col]['rule']
        Array[(f-1),(m),(d-1),1,1,0] = df_RT_profit[col]['det']
        Array[(f-1),(m),(d-1),2,1,0] = df_RT_profit[col]['sto']
        Array[(f-1),(m),(d-1),3,1,0] = df_RT_profit[col]['feature']
        Array[(f-1),(m),(d-1),4,1,0] = df_RT_profit[col]['oracle']
        
    return Array


def Count_performance_for_each_model(Array,includeOracle = False):
    
    #  Array = np.empty((6, 1, 88, 5, 2))
    #  (number of different forecast accuracies,number of different train sizes, test days, models, result) with empty values

    # How many times does each model win?
    
    # Pick Oracle or not?
    if includeOracle == False:
        noOracle = 4
    else:
        noOracle = 5

    model_order = ['rule','det','sto','feature','oracle']
    profit_type_order = ['Expected','RT']
    Array_to_use = Array[:,:,:,0:noOracle,:]

    CountArray = np.zeros((Array_to_use.shape[0],Array_to_use.shape[1],Array_to_use.shape[2],Array_to_use.shape[3],Array_to_use.shape[4]), dtype=int)  # Initialize the counts matrix
    for f in range(Array.shape[0]): # Forecast
        for sample_n in range(Array.shape[1]): # Sample length
            for profit in range(Array.shape[4]): # Expected / real profit
                for profit_type in range(2):
                    for t in range(0,88):
                        max_value = np.max(Array_to_use[f, sample_n, t, :, profit_type])
                        max_index = np.argmax(Array_to_use[f, sample_n, t, :, profit_type]) 

                        CountArray[f, sample_n, t, max_index, profit_type] = 1


    # CountArray_new has dim (f, sample_len, profit_type, model_type)
    return CountArray, model_order,profit_type_order

def plot_each_test_day_Profit(data,models,x_axis,drawstyle,save = False):
    #x_axis = np.arange(88)
    color_models = ['C3','C0', 'C2', 'C1','C4']
    model_labels = ['Rule','Deterministic','Stochastic','Feature','Oracle']
    y_axis_label = 'Realized Profit [\u20AC/day]'
    plot = data[models]

    
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)

    for m,model in enumerate(models):
        if model == 'oracle':
            ax.plot(x_axis,data[model], 'o-',color=color_models[m],drawstyle=drawstyle,markersize=10, linewidth=1,linestyle='--', marker='o')
        else:
            ax.plot(x_axis,data[model], 'o-',color=color_models[m],drawstyle=drawstyle,markersize=10, linewidth=1)
    # Set the number of decimals for the y-axis tick labels
    formatter = ScalarFormatter(useOffset=False,useMathText=True)
    formatter.set_powerlimits((-1, 2))
    ax.yaxis.set_major_formatter(formatter)
    # Adjust the font size of the exponent notation
    ax.yaxis.get_offset_text().set_fontsize(16)  # Set the font size
    # Increase the size of the y-axis ticks
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)  # Set the font size

    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)

    ax.set_ylabel(y_axis_label, fontsize=22, labelpad=10)
    ax.set_xlabel('Test day', fontsize=18, labelpad=10)


    ax.legend(model_labels, loc='lower center', ncol=len(color_models), bbox_to_anchor=(0.5, -0.185),prop={'size': 20})

    if save == True:
        plt.savefig(f'Result_plots/Profit_all_test_days_plot.pdf',format="pdf", bbox_inches='tight')

    plt.show()
    

def plot_profit_Test(Array, PlotCase = "", Selected_models = ['Rule', 'Deterministic', 'Stochastic', 'Feature','Oracle'], x_axis_label = "Forecasts", x_axis_selection = [0,1,2], x_axis_tick_label = [1,2,3], 
                     y_axis_label =  'Profit [\u20AC/day]', Selected_Profit = ['Expected', 'Realized'], 
                     barwidth = 0.1, ylim = "",
                     bbox_to_anchor=(1.02, 0.9),legends = ['Rule', 'Deterministic', 'Stochastic', 'Feature','Oracle','Expected','Realized'], 
                     ShowEachTestDay = False, pdf_name = 'Profit_Bar_plot' ,save = False):

    # Sample data
    #Array = np.random.rand(3, 6, 88, 5, 2)
    # PlotCase = 'Forecast' , 'Sample Size', ''
    #Forecast_names = [1,2,3]
    #SampleSize_names = [2,5,7,9,11]
    
    
    # Default settings
    color = ['C3', 'C0', 'C2', 'C1']  # Colors for the fourth dimension
    Model = ['Rule', 'Deterministic', 'Stochastic', 'Feature']
    Profit_labels = ['Expected', 'Realized']  # Fill patterns for the fifth dimension
    Profit_patterns = ['', '\\']  # Fill patterns for the fifth dimension
    # /   - diagonal hatching
    # \   - back diagonal
    # |   - vertical
    # -   - horizontal
    # +   - crossed
    # x   - crossed diagonal
    # o   - small circle
    # O   - large circle
    # .   - dots
    # *   - stars
    bar_count = (len(Selected_models)) * len(Selected_Profit) # Number of bars in each group (5-1)*2
    if "Oracle" in Selected_models:
        linestyle = '--'
        linecolor = 'C4'
        linelabel = 'Oracle'
        bar_count = (len(Selected_models)-1) * len(Selected_Profit) # Number of bars in each group (5-1)*2
        

    # From Selected_models and Selected_Profit specify necessary settings aligned with the default
    model_idx = [i for i, m in enumerate(Model) if m in Selected_models]
    models = [Model[i] for i, m in enumerate(Model) if m in Selected_models]
    colors = [color[i] for i, m in enumerate(Model) if m in Selected_models]
    profit_idx = [i for i, p in enumerate(Profit_labels) if p in Selected_Profit]
    fill_patterns_label = [Profit_labels[i] for i, p in enumerate(Profit_labels) if p in Selected_Profit]
    fill_patterns = [Profit_patterns[i] for i, p in enumerate(Profit_labels) if p in Selected_Profit]  # Fill patterns for the fifth dimension
    
    mean_values = np.mean(Array, axis=2)
    std_values = np.std(Array, axis=2)

    if PlotCase == "":
        x_axis_label = '' 
        xtick_names = 'off'
        x_axis = [1]
        
    else:
        x_axis = np.arange(len(x_axis_selection))
        xtick_names = x_axis_tick_label

    # Fixed settings

    y_axis_label = y_axis_label
    test_days = np.arange(Array.shape[2])

    legend_labels = models
    legend_colors = colors

    edgecolor = "black"
    linewidth = 4
    
    # Build the bar sizing and spacing
    bar_list = [i*0 for i in range(bar_count)]
    spacing = barwidth/10
    for i in range(bar_count):
        if i == 0:
            bar_list[i] = -(barwidth * bar_count / 2) + i * barwidth
        else:
            bar_list[i] = bar_list[i-1] + barwidth
        
        if i % 2 == 0:
            bar_list[i] = bar_list[i] + spacing
        
    size_around_tick = barwidth * bar_count / 2 + barwidth/2   

    

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Build the oracle model:

    # Plot value of the oracle model
    if "Oracle" in Selected_models:
        mean_value = np.mean(Array[0, 0, :, 4, 1,0])
        lines = ax.hlines(mean_value, 0 - size_around_tick + 4 *spacing, len(x_axis) - 1 + size_around_tick, color=linecolor, linestyle=linestyle, linewidth=linewidth)


    for x,xaxis  in enumerate(x_axis_selection):
        count_j = 0
        count_k = 0
        for j, color in enumerate(colors):
            for k, fill_pattern in enumerate(fill_patterns):
                # construct where on the x axis the bar is located

                # To get a little bit of spacing after each model
                
                j_mod = model_idx[j]
                k_mod = profit_idx[k]

                if PlotCase == 'Forecasts':
                    bar_left = x + bar_list[count_k]
                    f = x
                    s = 0
                    m = 0   

                elif PlotCase == 'Sample size':
                    bar_left = x + bar_list[count_k]
                    f = 0
                    s = x
                    m = 0
                
                elif PlotCase == 'Model Exploration':
                    bar_left = x + bar_list[count_k]
                    f = 0
                    s = 0
                    m = x

                else:
                    bar_left = bar_list[count_k]
                    f = 0
                    s = 0
                    m = 0


                
                # Define the height of the bar
                bar_height = mean_values[f,s,j_mod,k_mod,m]
                error_value = std_values[f,s,j_mod,k_mod,m]

                

                ax.bar(bar_left, bar_height, width=barwidth, color=color, hatch=fill_pattern, align='edge', edgecolor=edgecolor)

                x_position = bar_left + barwidth/2
                ax.errorbar(x_position, bar_height, yerr=error_value, fmt='none', ecolor='black', capsize=4)

                if ShowEachTestDay == True:
                    for d,val in enumerate(test_days):
                        #print("(",bar_left,",",Array[x,2,d,j,k],")")
                        
                        ax.scatter(x_position,Array[f,s,j_mod,k_mod],s=2, edgecolor=edgecolor, color=color)


                count_k = count_k + 1
            count_j = count_j + 1

    if ylim != "":
        ax.set_ylim(ylim[0], ylim[1])

    # Set the x-axis ticks and tick labels
    
    if xtick_names != 'off':
        ax.set_xticks(x_axis)
        ax.set_xticklabels(xtick_names, fontsize=16)
        ax.set_xlabel(x_axis_label, fontsize=16, labelpad=10)
    else:
        # Remove x axis
            
        #ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')

    # Set the y-axis tick size
    ax.tick_params(axis='y', which='major', labelsize=20)
    
    ax.set_ylabel(y_axis_label, fontsize=16, labelpad=10)
    # Set the y-axis tick formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 2))  # Adjust the power limits as needed
    plt.gca().yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(16)  # Set the font size
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)

    # Create legend for the bars
    legend_handles = []

    for label, color in zip(legend_labels, legend_colors):
        if label in legends:
            rect = plt.Rectangle((0, 0), 1, 1, color=color, label=label)
            rect.set_edgecolor(edgecolor)
            legend_handles.append(rect)



    for label, pattern in zip(fill_patterns_label, fill_patterns):
        if label in legends: 
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=edgecolor,  hatch=pattern, label=label))

    if "Oracle" in Selected_models:
        # Create legend for the horizontal line
        if "Oracle" in legends:
            legend_handles.append(plt.Line2D([0], [0], color=linecolor, linestyle=linestyle, label=linelabel,linewidth=linewidth))
    
    # Show the legend
    ax.legend(handles=legend_handles,bbox_to_anchor=bbox_to_anchor, loc='upper left',prop={'size': 14})



    if save == True:
        plt.savefig(f'Result_plots/'+ pdf_name +'.pdf',format="pdf", bbox_inches='tight')
    # Show the plot
    plt.show()



def Collect_bid_results(current_directory,Add_on_path,forecast_idx,SampleSize_idx,Acceptance_idx):
    
    #forecast_idx = 1
    #SampleSize_idx = 5
    #acceptance_criteria_factor_range = ["1.00","1.05","1.1","1.2"]
    acceptance_count = ["acceptance_FD2_down","acceptance_FD2_up","acceptance_FD1_down","acceptance_FD1_up"]
    bid_quantities   = ["b_FD2_dn","b_FD2_up","b_DA_dn","b_DA_up","b_FD1_dn","b_FD1_up"]
    models = ["Rule","Det","Sto","Feature","Oracle"]
    Array_Acceptance = np.empty((len(acceptance_count),24, 88, 5,len(Acceptance_idx)))
    Array_Bid = np.empty((len(bid_quantities),24, 88, 5))
    
    for ac,acc in enumerate(Acceptance_idx):
        if len(Acceptance_idx) > 1:
            ids = [f'acc_{acc}_f{forecast_idx}_d{SampleSize_idx}_upd{d}_t{d+1}' for d in range(0,88)]
        else:
            ids = [f'f{forecast_idx}_d{SampleSize_idx}_upd{d}_t{d+1}' for d in range(0,88)]



        # Load each id at a time
        for d,id in enumerate(ids):
            results = import_test_case(current_directory, Add_on_path, id)
            #print("Import ", id)


            # Gather all the information:
            for m,model in enumerate(models):
                # Acceptance
                for a,acc in enumerate(acceptance_count):
                    #print(results[model]['RT'][acc])
                    Array_Acceptance[a,:,d,m,ac] = results[model]['RT'][acc]

                # Bid quantity
                for b,bid in enumerate(bid_quantities):
                    #print(results[model]['Bid'][bid])
                    Array_Bid[b,:,d,m] = results[model]['Bid'][bid]
    
    return Array_Acceptance, Array_Bid



def Plot_average_bid_quantities(Array_Acceptance,Array_Bid):
    Acceptance_mean = np.mean(Array_Acceptance, axis=2)
    Acceptance_std = np.std(Array_Acceptance, axis=2)
    Bid_mean = np.mean(Array_Bid, axis=2)
    Bid_std = np.std(Array_Bid, axis=2)

    x = np.arange(1, 25)

    # Determine the number of subplots based on b iterations
    bid_quantities = ["b_FD2_dn", "b_FD2_up", "b_DA_dn", "b_DA_up", "b_FD1_dn", "b_FD1_up"]
    bid_quantities_label = ["$FD2\\downarrow$", "$FD2\\uparrow$", "$DA\\downarrow$", "$DA\\uparrow$", "$FD1\\downarrow$", "$FD1\\uparrow$"]

    models = ["Rule", "Deterministic", "Stochastic", "Feature", "Oracle"]
    model_colors = ['C3','C0', 'C2', 'C1','C4']
    linestyles = ['-','-','-','-','--']
    num_subplots = len(bid_quantities)

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))

    # Iterate over each bid and create a subplot
    for b, bid in enumerate(bid_quantities):
        row = b // 2  # Determine the row index
        col = b % 2  # Determine the column index

        for m, model in enumerate(models):
            axs[row, col].plot(x, Bid_mean[b, :, m], label=model,color=model_colors[m], linestyle = linestyles[m],linewidth=2.5)
            # axs[row, col].scatter(x, Acceptance_mean[1, :, m], label=model)  # Uncomment if needed

        axs[row, col].set_title(bid_quantities_label[b], fontsize=16)  # Set subplot title
        #axs[row, col].legend()

        # Remove x-axis for subplots that are not in the last row
        if row != axs.shape[0] - 1:
            axs[row, col].set_xticks([])
            axs[row, col].set_xlabel('')
        else:
            axs[row, col].set_xticks([1,5,10,15,20,24])
            axs[row, col].set_xlabel('Time [h]')

        # Set y-axis label for subplots in the first column
        if col == 0:
            axs[row, col].set_ylabel('Average bid quantity [MW]')

        # Set the number of decimals for the y-axis tick labels
        formatter = ScalarFormatter(useOffset=False,useMathText=True)
        formatter.set_powerlimits((-1, 2))
        axs[row, col].yaxis.set_major_formatter(formatter)
        axs[row, col].yaxis.set_major_formatter('{x:.2f}')  # Set 2 decimals, adjust as needed

            # Set the y-axis tick formatter
        #formatter = ScalarFormatter(useMathText=True)
        #formatter.set_powerlimits((-1, 2))  # Adjust the power limits as needed
        #plt.gca().yaxis.set_major_formatter(formatter)

        
    # Remove unused subplots if there are fewer bid quantities than the total number of subplots
    if num_subplots < axs.size:
        for i in range(num_subplots, axs.size):
            axs.flat[i].set_visible(False)

    # Create a common legend outside the subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, -0.03),
                        prop={'size': 16})  # Adjust the fontsize of the legend here


    # Adjust the fontsize of the axis ticks and labels
    for ax in axs.flat:
        ax.tick_params(axis='both', which='both', labelsize=14)  # Adjust the fontsize of the axis ticks here
        ax.xaxis.label.set_fontsize(16)  # Adjust the fontsize of the x-axis label here
        ax.yaxis.label.set_fontsize(16)  # Adjust the fontsize of the y-axis label here



    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig('Bid_quantities.pdf', format="pdf", bbox_inches="tight")
    plt.show()


def plot_Profit_Test_For_Forecast(Array, xtick_names,SampleSize_index, marker_size = 2, marker_size_2 = 20):

    Array = Array[:,SampleSize_index,:,:,:]
    iteration_numbers = Array.shape[0]
    x_length = Array.shape[0]
    xlabel_name = 'Forecasts'
    ax1_name = 'Profit for different forecasts and at different test days'
    ax2_name = 'Mean profit for different forecasts'
    xtick_range = range(0, x_length)
        
    # Set up the plot
    #fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    # Define a list of colors for models
    colors = ['red' , 'blue', 'green', 'orange','purple']
    Model  = ['Rule', 'Det' , 'Sto'  , 'Feature' ,'oracle']

    # Define a list of symbols for profits
    symbols = ['o'  , 's' ]
    Profit     = ['Exp', 'RT']

    # Define a list of x-axis movement for each profit
    xaxis_move = [-0.25,-0.2,-0.15,-0.1,-0.05, 0.05,0.1,0.15,0.2,0.25]



    # Calculate the mean along the test day axis (axis=1)
    mean_array_along_d = np.mean(Array, axis=1, keepdims=True)

    # Iterate over d values
    for d in range(iteration_numbers):
        # Iterate over profit values
        count = 0
        for profit in range(len(Profit)):
            # Iterate over model values
            for model in range(len(Model)):
                

                data = Array[:, d, model, profit]

                # Get the unique color and symbol based on the model and profit
                color = colors[model]
                symbol = symbols[profit]

                # Calculate the modified x values based on the profit
                
                modified_x = list(range(x_length))  # Convert range object to a list
                modified_x = [x + xaxis_move[count] for x in modified_x]
                count = count + 1 # For modified_x

                # Plot the data with the unique color and symbol
                ax1.scatter(modified_x, data, color=color, marker=symbol,s=marker_size)


    # Set y-axis label
    ax1.set_ylabel('Profit [EUR]')

    # Create a legend for colors and symbols
    legend_handles = []
    for model, color in enumerate(colors):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f'{Model[model]}', markerfacecolor=color, markersize=10))
    for profit, symbol in enumerate(symbols):
        legend_handles.append(Line2D([0], [0], marker=symbol, color='w', label=f'{Profit[profit]}', markerfacecolor='black', markersize=10))

    # Add the legend to the plot outside the graph area
    ax1.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 0.7))

    # The other subplot
    count = 0
    for profit in range(len(Profit)):
        # Iterate over model values
        for model in range(len(Model)):

            # Extract the data for the current d, model, and profit
            data = mean_array_along_d[:, 0, model, profit]

            # Get the unique color and symbol based on the model and profit
            color = colors[model]
            symbol = symbols[profit]

            # Calculate the modified x values based on the profit
            modified_x = list(range(x_length))  # Convert range object to a list
            modified_x = [x + xaxis_move[count] for x in modified_x]
            count = count + 1  # For modified_x

            # Plot the data with the unique color and symbol
            #if model == 0 or model == 1 or model == 4: # if rule, det or 
            ax2.scatter(modified_x, data, color=color, marker=symbol, s=marker_size_2)


    # Set y-axis label
    ax2.set_ylabel('Profit [EUR]')

    # Set x-axis label
    ax2.set_xlabel(xlabel_name)
    ax2.set_xticks(xtick_range,xtick_names) 

    # Set title for ax1
    ax1.set_title(ax1_name)

    # Set title for ax2
    ax2.set_title(ax2_name)

    # Show the plot
    plt.show()



def plot_Profit_Test_For_SampleSize(Array, xtick_names,Forecast_index, marker_size = 2, marker_size_2 = 20):
    # Assuming Array[f, m,d,model,profit] is the four-dimensional array
    Array = Array[Forecast_index,:,:,:,:]
    iteration_numbers = Array.shape[0]
    x_length = Array.shape[0]
    xlabel_name = 'Training Sample'
    ax1_name = 'Profit for different training sizes and at different test days'
    ax2_name = 'Mean profit for different training sizes'
    xtick_range = range(0, x_length)

        
    # Set up the plot
    #fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    # Define a list of colors for models
    colors = ['red' , 'blue', 'green', 'orange','purple']
    Model  = ['Rule', 'Det' , 'Sto'  , 'Feature' ,'oracle']

    # Define a list of symbols for profits
    symbols = ['o'  , 's' ]
    Profit     = ['Exp', 'RT']

    # Define a list of x-axis movement for each profit
    xaxis_move = [-0.25,-0.2,-0.15,-0.1,-0.05, 0.05,0.1,0.15,0.2,0.25]



    # Calculate the mean along the test day axis (axis=1)
    mean_array_along_d = np.mean(Array, axis=1, keepdims=True)

    # Iterate over d values
    for d in range(iteration_numbers):
        # Iterate over profit values
        count = 0
        for profit in range(len(Profit)):
            # Iterate over model values
            for model in range(len(Model)):
                

                data = Array[:, d, model, profit]

                # Get the unique color and symbol based on the model and profit
                color = colors[model]
                symbol = symbols[profit]

                # Calculate the modified x values based on the profit
                
                modified_x = list(range(x_length))  # Convert range object to a list
                modified_x = [x + xaxis_move[count] for x in modified_x]
                count = count + 1 # For modified_x

                # Plot the data with the unique color and symbol
                ax1.scatter(modified_x, data, color=color, marker=symbol,s=marker_size)


    # Set y-axis label
    ax1.set_ylabel('Profit [EUR]')

    # Create a legend for colors and symbols
    legend_handles = []
    for model, color in enumerate(colors):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f'{Model[model]}', markerfacecolor=color, markersize=10))
    for profit, symbol in enumerate(symbols):
        legend_handles.append(Line2D([0], [0], marker=symbol, color='w', label=f'{Profit[profit]}', markerfacecolor='black', markersize=10))

    # Add the legend to the plot outside the graph area
    ax1.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.1, 0.7))

    # The other subplot
    count = 0
    for profit in range(len(Profit)):
        # Iterate over model values
        for model in range(len(Model)):

            # Extract the data for the current d, model, and profit
            data = mean_array_along_d[:, 0, model, profit]

            # Get the unique color and symbol based on the model and profit
            color = colors[model]
            symbol = symbols[profit]

            # Calculate the modified x values based on the profit
            modified_x = list(range(x_length))  # Convert range object to a list
            modified_x = [x + xaxis_move[count] for x in modified_x]
            count = count + 1  # For modified_x

            # Plot the data with the unique color and symbol
            #if model == 0 or model == 1 or model == 4: # if rule, det or 
            ax2.scatter(modified_x, data, color=color, marker=symbol, s=marker_size_2)


    # Set y-axis label
    ax2.set_ylabel('Profit [EUR]')

    # Set x-axis label
    ax2.set_xlabel(xlabel_name)
    ax2.set_xticks(xtick_range,xtick_names) 

    # Set title for ax1
    ax1.set_title(ax1_name)

    # Set title for ax2
    ax2.set_title(ax2_name)

    # Show the plot
    plt.show()


def import_test_case(current_directory, Add_on_path, choose_id):
    #Import JSON files
    with open(current_directory + Add_on_path + f'/oracle_{choose_id}.json') as results_oracle_json:
        Results_oracle = json.load(results_oracle_json)
    with open(current_directory + Add_on_path + f'/det_{choose_id}.json') as results_det_json:
        Results_det = json.load(results_det_json)
    with open(current_directory + Add_on_path + f'/feature_{choose_id}.json') as results_feature_json:
        Results_feature = json.load(results_feature_json)
    with open(current_directory + Add_on_path + f'/sto_{choose_id}.json') as results_sto_json:
        Results_sto = json.load(results_sto_json)
    with open(current_directory + Add_on_path + f'/rule_{choose_id}.json') as results_rule_json:
        Results_rule = json.load(results_rule_json)

    #Convert results in to np.arrays
    oracle_bid = {}
    oracle_RT = {}
    det_bid = {}
    det_RT = {}
    feature_bid = {}
    feature_RT = {}
    sto_bid = {}
    sto_RT = {}
    rule_bid = {}
    rule_RT = {}

    for key, value in Results_oracle['Bid'].items():
        oracle_bid[key] = np.array(value)
    for key, value in Results_oracle['RT'].items():
        oracle_RT[key] = np.array(value)

    for key, value in Results_det['Bid'].items():
        det_bid[key] = np.array(value)
    for key, value in Results_det['RT'].items():
        det_RT[key] = np.array(value)

    for key, value in Results_feature['Bid'].items():
        feature_bid[key] = np.array(value)
    for key, value in Results_feature['RT'].items():
        feature_RT[key] = np.array(value)

    for key, value in Results_sto['Bid'].items():
        sto_bid[key] = np.array(value)
    for key, value in Results_sto['RT'].items():
        sto_RT[key] = np.array(value)

    for key, value in Results_rule['Bid'].items():
        rule_bid[key] = np.array(value)
    for key, value in Results_rule['RT'].items():
        rule_RT[key] = np.array(value)

    results = {'Oracle': {'Bid': oracle_bid, 'RT': oracle_RT},
               'Det': {'Bid':det_bid, 'RT':det_RT}, 
               'Rule': {'Bid':rule_bid, 'RT':rule_RT}, 
               'Feature': {'Bid':feature_bid, 'RT':feature_RT}, 
               'Sto': {'Bid':sto_bid, 'RT':sto_RT}}
    return results


def bid_plots(bid_result, model, color, save):
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
    # ax1.set_xticks([1, 6, 12, 18, 24])
    ax1.set_xticks(x)
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

    if save == True:
        plt.savefig(f'Result_plots/bid_plots_{model}.png', bbox_inches='tight')

    # plt.show()


    

def plot_training_price(with_acceptance, bid_result, model, color, save):
    # Plot prices used for training
    fig, ax2 = plt.subplots(figsize=(7,5))

    x = np.arange(1,25)
    if 'Rule' in model:
        ax2.plot(x, bid_result['f_lambda_FD2_up'], label='f_FD2_up', marker='.', color=color['FD2_up'])
        ax2.plot(x, bid_result['f_lambda_FD2_dn'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
        ax2.plot(x, bid_result['f_DA_t'], label='f_DA', marker='.', color=color['DA_up']) #Expected DA price
        ax2.plot(x, bid_result['Threshold_Max'], linestyle='--', color='grey')
        ax2.plot(x, bid_result['Threshold_Min'], linestyle='--', color='grey')
        ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)

    elif 'Det' in model:
        ax2.plot(x, bid_result['f_DA_t'], label='f_DA', marker='.', color=color['DA_up']) #Expected DA price
        if with_acceptance == True:
            ax2.plot(x, bid_result['f_lambda_FD2_up']*bid_result['f_FD2_y_up_t'], label='f_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, bid_result['f_lambda_FD2_dn']*bid_result['f_FD2_y_dn_t'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
            ax2.plot(x, bid_result['f_lambda_FD1_up']*bid_result['f_FD1_y_up_t'], label='f_FD1_up', marker='.', color=color['FD1_up'])
            ax2.plot(x, bid_result['f_lambda_FD1_dn']*bid_result['f_FD1_y_dn_t'], label='f_FD1_dn', marker='.', color=color['FD1_dn'])
            ax2.set_ylabel('Price used for training with acceptance [EUR/MW]', fontsize=12)
        elif with_acceptance == False:
            ax2.plot(x, bid_result['f_lambda_FD2_up'], label='f_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, bid_result['f_lambda_FD2_dn'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
            ax2.plot(x, bid_result['f_lambda_FD1_up'], label='f_FD1_up', marker='.', color=color['FD1_up'])
            ax2.plot(x, bid_result['f_lambda_FD1_dn'], label='f_FD1_dn', marker='.', color=color['FD1_dn'])
            ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)            
    elif 'Oracle' in model:
        ax2.plot(x, bid_result['f_DA_t'], label='f_DA', marker='.', color=color['DA_up']) #Expected DA price
        ax2.plot(x, bid_result['f_lambda_FD2_up'], label='f_FD2_up', marker='.', color=color['FD2_up'])
        ax2.plot(x, bid_result['f_lambda_FD2_dn'], label='f_FD2_dn', marker='.', color=color['FD2_dn'])
        ax2.plot(x, bid_result['f_lambda_FD1_up'], label='f_FD1_up', marker='.', color=color['FD1_up'])
        ax2.plot(x, bid_result['f_lambda_FD1_dn'], label='f_FD1_dn', marker='.', color=color['FD1_dn'])
        ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)  

    elif 'Feature' in model: # This only plots training data for now, the bid price is equal to the forecast price
        X_train_spot = np.array(bid_result["X"])[0,:,:].T
        X_train_FD1_dn = np.array(bid_result["X"])[1,:,:].T
        X_train_FD2_dn = np.array(bid_result["X"])[2,:,:].T
        X_train_FD1_up = np.array(bid_result["X"])[3,:,:].T
        X_train_FD2_up = np.array(bid_result["X"])[4,:,:].T

        ax2.plot(x, X_train_spot.mean(axis=1), label='X_spot', marker='.', color=color['DA_up'])
        if with_acceptance == True:
            ax2.plot(x, X_train_FD2_up.mean(axis=1)*bid_result['f_FD2_y_up_t'].flatten(), label='X_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, X_train_FD2_dn.mean(axis=1)*bid_result['f_FD2_y_dn_t'].flatten(), label='X_FD2_dn', marker='.', color=color['FD2_dn'])
            ax2.plot(x, X_train_FD1_up.mean(axis=1)*bid_result['f_FD1_y_up_t'].flatten(), label='X_FD1_up', marker='.', color=color['FD1_up'])
            ax2.plot(x, X_train_FD1_dn.mean(axis=1)*bid_result['f_FD1_y_dn_t'].flatten(), label='X_FD1_dn', marker='.', color=color['FD1_dn'])
            ax2.set_ylabel('Price used for training with acceptance [EUR/MW]', fontsize=12)

        elif with_acceptance == False:
            ax2.plot(x, X_train_FD2_up.mean(axis=1), label='X_FD2_up', marker='.', color=color['FD2_up'])
            ax2.plot(x, X_train_FD2_dn.mean(axis=1), label='X_FD2_dn', marker='.', color=color['FD2_dn'])
            ax2.plot(x, X_train_FD1_up.mean(axis=1), label='X_FD1_up', marker='.', color=color['FD1_up'])
            ax2.plot(x, X_train_FD1_dn.mean(axis=1), label='X_FD1_dn', marker='.', color=color['FD1_dn'])
            ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)

    elif 'Sto' in model:
        W_size = bid_result['f_DA_tw_input'].shape[0]

        FD2_up_mean = np.mean(bid_result['f_FD2_up_tw_input'], axis=0)
        FD2_up_max = np.max(bid_result['f_FD2_up_tw_input'], axis=0)
        FD2_up_min = np.min(bid_result['f_FD2_up_tw_input'], axis=0)

        FD2_dn_mean = np.mean(bid_result['f_FD2_dn_tw_input'], axis=0)
        FD2_dn_max = np.max(bid_result['f_FD2_dn_tw_input'], axis=0)
        FD2_dn_min = np.min(bid_result['f_FD2_dn_tw_input'], axis=0)

        DA_mean = np.mean(np.reshape(bid_result['f_DA_tw_input'], (W_size*W_size,24)), axis=0)
        DA_max = np.max(np.reshape(bid_result['f_DA_tw_input'], (W_size*W_size,24)), axis=0)
        DA_min = np.min(np.reshape(bid_result['f_DA_tw_input'], (W_size*W_size,24)), axis=0)

        FD1_up_mean = np.mean(np.reshape(bid_result['f_FD1_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_up_max = np.max(np.reshape(bid_result['f_FD1_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_up_min = np.min(np.reshape(bid_result['f_FD1_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)

        FD1_dn_mean = np.mean(np.reshape(bid_result['f_FD1_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_dn_max = np.max(np.reshape(bid_result['f_FD1_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_dn_min = np.min(np.reshape(bid_result['f_FD1_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)

        FD2_up_accept_mean = np.mean(bid_result['f_FD2_y_up_tw_input'], axis=0)
        FD2_up_accept_max = np.max(bid_result['f_FD2_y_up_tw_input'], axis=0)
        FD2_up_accept_min = np.min(bid_result['f_FD2_y_up_tw_input'], axis=0)

        FD2_dn_accept_mean = np.mean(bid_result['f_FD2_y_dn_tw_input'], axis=0)
        FD2_dn_accept_max = np.max(bid_result['f_FD2_y_dn_tw_input'], axis=0)
        FD2_dn_accept_min = np.min(bid_result['f_FD2_y_dn_tw_input'], axis=0)

        FD1_up_accept_mean = np.mean(np.reshape(bid_result['f_FD1_y_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_up_accept_max = np.max(np.reshape(bid_result['f_FD1_y_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_up_accept_min = np.min(np.reshape(bid_result['f_FD1_y_up_tw_input'], (W_size*W_size*W_size,24)), axis=0)

        FD1_dn_accept_mean = np.mean(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_dn_accept_max = np.max(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)
        FD1_dn_accept_min = np.min(np.reshape(bid_result['f_FD1_y_dn_tw_input'], (W_size*W_size*W_size,24)), axis=0)

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
            ax2.set_ylabel('Price used for training with acceptance [EUR/MW]', fontsize=12)
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
            ax2.set_ylabel('Price used for training [EUR/MW]', fontsize=12)

    ax2.set_xlabel('Hours', fontsize=12)
    # ax2.set_xticks([1, 6, 12, 18, 24])
    ax2.set_xticks(x)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.15, -0.15), ncol=3)
    ax2.set_title(model, fontsize=14)

    if save == True:
        plt.savefig(f'Result_plots/training_price_{model}.png', bbox_inches='tight')

    # plt.show()

def plot_bidding_price(bid_result, model, color, save):
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
    # ax3.set_xticks([1, 6, 12, 18, 24])
    ax3.set_xticks(x)
    ax3.legend(loc='upper left', bbox_to_anchor=(0.15, -0.15), ncol=3)
    ax3.set_title(model, fontsize=14)

    if save == True:
        plt.savefig(f'Result_plots/bidding_price_{model}.png', bbox_inches='tight')

    # plt.show()

def plot_accepted_price(data_RT_input, model, save):
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

    if save == True:
        plt.savefig(f'Result_plots/accepted_price_{model}.png', bbox_inches='tight')

    fig.suptitle(model, fontsize=12) #Remove this for overleaf export

def plot_coefficients(feature_bid, color, save):
    # Plot coefficients
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up", "1"]
    q_FD2_up = feature_bid["q_FD2_up"]
    q_FD2_dn = feature_bid["q_FD2_dn"]
    q_FD1_up = feature_bid["q_FD1_up"]
    q_FD1_dn = feature_bid["q_FD1_dn"]
    q_DA_up = feature_bid["q_DA_up"]
    q_DA_dn = feature_bid["q_DA_dn"]

    fig, ax4 = plt.subplots(figsize=(7,5))

    ax4.set_ylabel('Coefficients', fontsize=12)
    ax4.set_xlabel('Features', fontsize=12)
    ax4.plot(q_FD2_up, label='q_FD2_up', marker = '.', color=color['FD2_up'])
    ax4.plot(q_FD2_dn, label='q_FD2_dn', marker = '.', color=color['FD2_dn'])
    ax4.plot(q_FD1_up, label='q_FD1_up', marker = '.', color=color['FD1_up'])
    ax4.plot(q_FD1_dn, label='q_FD1_dn', marker = '.', color=color['FD1_dn'])
    ax4.plot(q_DA_up, label='q_DA_up', marker = '.', color=color['DA_up'])
    ax4.plot(q_DA_dn, label='q_DA_dn', marker = '.', color=color['DA_dn'])
    ax4.set_xticks([i for i in range(0,len(q_FD2_up))], Feature_Selection, rotation=40)
    ax4.legend(loc='upper left', bbox_to_anchor=(0.2, -0.25), ncol=2, fontsize=12)

    if save == True:
        plt.savefig('Result_plots/coefficients.png', bbox_inches='tight')

    # plt.show()

def plot_exp_and_RT_profit(results, profit_plot, save):
    # Plot RT and expected profit
    fig, ax6 = plt.subplots(figsize=(7,5))
    x = np.arange(1,25)

    if 'Det' in profit_plot:
        ax6.plot(x, results['Det']['RT']['profit_t'], label='Det_RT', marker = '.', color='C0') 
        ax6.plot(x, results['Det']['Bid']['obj_t'], label='Det_Exp', linestyle='dashed', color='C0') 
    elif 'Sto' in profit_plot:
        ax6.plot(x, results['Sto']['RT']['profit_t'], label='Sto_RT', marker = '.', color='C1') 
        ax6.plot(x, results['Sto']['Bid']['obj_t'], label='Sto_Exp', linestyle='dashed', color='C1') 
    elif 'Feature' in profit_plot:
        ax6.plot(x, results['Feature']['RT']['profit_t'], label='Feature_RT', marker = '.', color='C2') 
        ax6.plot(x, results['Feature']['Bid']['obj_t'].flatten(), label='Feature_Exp', linestyle='dashed', color='C2') 
    elif 'Rule' in profit_plot:
        ax6.plot(x, results['Rule']['RT']['profit_t'], label='Rule_RT', marker = '.', color='C3') 
        ax6.plot(x, results['Rule']['Bid']['obj_t'], label='Rule_Exp', linestyle='dashed', color='C3') 
    elif 'Oracle' in profit_plot:
        ax6.plot(x, results['Oracle']['RT']['profit_t'], label='Rule_RT', marker = '.', color='C4') 
        ax6.plot(x, results['Oracle']['Bid']['obj_t'], label='Rule_Exp', linestyle='dashed', color='C4') 
    elif 'all' in profit_plot:
        ax6.plot(x, results['Det']['RT']['profit_t'], label='Det_RT', marker = '.', color='C0') 
        ax6.plot(x, results['Det']['Bid']['obj_t'], label='Det_Exp', linestyle='dashed', color='C0') 
        ax6.plot(x, results['Sto']['RT']['profit_t'], label='Sto_RT', marker = '.', color='C1') 
        ax6.plot(x, results['Sto']['Bid']['obj_t'], label='Sto_Exp', linestyle='dashed', color='C1') 
        ax6.plot(x, results['Feature']['RT']['profit_t'], label='Feature_RT', marker = '.', color='C2') 
        ax6.plot(x, results['Feature']['Bid']['obj_t'].flatten(), label='Feature_Exp', linestyle='dashed', color='C2') 
        ax6.plot(x, results['Rule']['RT']['profit_t'], label='Rule_RT', marker = '.', color='C3') 
        ax6.plot(x, results['Rule']['Bid']['obj_t'], label='Rule_Exp', linestyle='dashed', color='C3') 
        ax6.plot(x, results['Oracle']['RT']['profit_t'], label='Rule_RT', marker = '.', color='C4') 
        ax6.plot(x, results['Oracle']['Bid']['obj_t'], label='Rule_Exp', linestyle='dashed', color='C4') 

    ax6.set_ylabel('Profit [EUR]', fontsize=12)
    ax6.set_xlabel('Hours', fontsize=12)
    # ax6.set_xticks([1,6,12,18,24])
    ax6.set_xticks(x)

    if 'all' in profit_plot:
        ax6.legend(loc='upper left', bbox_to_anchor=(-0.05, -0.15), ncol=4, fontsize=12)
    else:
        ax6.legend(loc='upper left', bbox_to_anchor=(0.15, -0.15), ncol=2, fontsize=12)

    if save == True:
        plt.savefig(f'Result_plots/exp_and_RT_profit_{profit_plot}.png', bbox_inches='tight')
    # plt.show()

def plot_battery_dynamics(RT_result, model, save):
    # Plot battery dynamics
    fig, ax5 = plt.subplots(figsize=(7, 5))
    x = np.arange(1,25)

    ax5.plot(x, RT_result['p_all_up'], label='p_all_up', marker = '.')
    ax5.plot(x, RT_result['p_all_dn'], label='p_all_dn', marker = '.')
    ax5.plot(x, RT_result['failure_up'], label='failure_up')
    ax5.plot(x, RT_result['failure_dn'], label='failure_dn')
    ax5.plot(x, RT_result['p_dis_max']*np.ones(24), '--' ,label='p_dis_max', color='grey')
    ax5.plot(x, RT_result['p_ch_max']*np.ones(24), '--' ,label='p_ch_max', color='grey')
    # ax5.legend(loc='center left', bbox_to_anchor=(0.15,-0.2), ncol=3)
    ax5.set_xlabel('Hours')
    ax5.set_ylabel('Power [MW]')
    # ax5.set_xticks([1,6,12,18,24])
    ax5.set_xticks(x)

    ax5_1 = ax5.twinx()
    ax5_1.plot(x, RT_result['SOC']/6, color='black', label='SOC', marker = '.')
    ax5_1.set_ylabel('SOC [p.u.]', fontsize=12)
    ax5_1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax5_1.set_ylim([-0.05,1.05])

    # Combine the legends
    handles1, labels1 = ax5.get_legend_handles_labels()
    handles2, labels2 = ax5_1.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, -0.15), ncol=4)

    if save == True:
        plt.savefig(f'Result_plots/battery_dynamics_{model}.png', bbox_inches='tight')

    # plt.show()

def save_plots(current_directory,Add_on_path, choose_id, save, model, with_acceptance, profit_plot, print_table):
    #Fix color schemes
    color = {'FD2_up': '#FFA500', # orange
            'FD2_dn': '#2986cc', # blue
            'DA_up': '#f44336', # red 
            'DA_dn': 'green', # green
            'FD1_up': '#b26d02', # brown orange
            'FD1_dn': '#674ea7', # darkgreen
            'SOC': 'black'}

    results = import_test_case(current_directory, Add_on_path, choose_id)

    #Plotting functions
    bid_plots(results[model]['Bid'], model, color, save)
    plot_training_price(with_acceptance, results[model]['Bid'], model, color, save)
    plot_bidding_price(results[model]['Bid'], model, color, save)
    plot_accepted_price(results[model]['RT'], model, save)
    if 'Feature' in model:
        plot_coefficients(results[model]['Bid'], color, save)
    plot_exp_and_RT_profit(results, profit_plot, save)
    plot_battery_dynamics(results[model]['RT'], model, save)
    
    #Print summary
    result_summary = [["RT", results['Oracle']['RT']['profit'],results['Rule']['RT']['profit'], results['Det']['RT']['profit'], results['Sto']['RT']['profit'], results['Feature']['RT']['profit']],
                      ["Expected", sum(results['Oracle']['Bid']['obj_t']),sum(results['Rule']['Bid']['obj_t']), sum(results['Det']['Bid']['obj_t']), sum(results['Sto']['Bid']['obj_t'].flatten()), sum(results['Feature']['Bid']['obj_t'].flatten())]]
    headers = ["Oracle","Rule", "Deterministic", "Stochastic", "Feature"]
    table = tabulate(result_summary, headers, tablefmt="grid")
    if print_table == 'yes':
        print('Test case: ', choose_id)
        print(table)

    return results

def view_plots(model, profit_plot):
    # Select plots
    image_paths = [f'Result_plots/bid_plots_{model}.png', 
                f'Result_plots/training_price_{model}.png', 
                f'Result_plots/bidding_price_{model}.png',
                f'Result_plots/exp_and_RT_profit_{profit_plot}.png',
                f'Result_plots/battery_dynamics_{model}.png']
    if 'Feature' in model:
        image_paths.append(f'Result_plots/coefficients.png')

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten() if 'Feature' in model else axes.flatten()[0:5]):
        image = Image.open(image_paths[i])
        ax.imshow(image)
        ax.axis("off")  # Remove the axis labels
            
        # Adjust the spacing between subplots
        plt.tight_layout()
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 10))
    image = Image.open(f'Result_plots/accepted_price_{model}.png')
    ax1.imshow(image)
    ax1.axis("off")
    plt.show()

#Heat map of hourly coefficients
def plot_coefficients_heatmap(feature_bid, save):
    # Plot coefficients
    x = np.arange(1,25)
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up", "1"]
    q_FD2_up = feature_bid["q_FD2_up"]
    q_FD2_dn = feature_bid["q_FD2_dn"]
    q_FD1_up = feature_bid["q_FD1_up"]
    q_FD1_dn = feature_bid["q_FD1_dn"]
    q_DA_up = feature_bid["q_DA_up"]
    q_DA_dn = feature_bid["q_DA_dn"]


    fig, ax = plt.subplots(6,1,figsize=(20,20))

    im = ax[0].imshow(q_FD2_up)
    ax[0].set_xticks(np.arange(len(x)), labels=x)
    ax[0].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[0].text(j, i, q_FD2_up[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[0].set_title("q_FD2_up")
    fig.tight_layout()

    im = ax[1].imshow(q_FD2_dn)
    ax[1].set_xticks(np.arange(len(x)), labels=x)
    ax[1].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[1].text(j, i, q_FD2_dn[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[1].set_title("q_FD2_dn")
    fig.tight_layout()

    im = ax[2].imshow(q_FD1_up)
    ax[2].set_xticks(np.arange(len(x)), labels=x)
    ax[2].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[2].text(j, i, q_FD1_up[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[2].set_title("q_FD1_up")
    fig.tight_layout()

    im = ax[3].imshow(q_FD1_dn)
    ax[3].set_xticks(np.arange(len(x)), labels=x)
    ax[3].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[3].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[3].text(j, i, q_FD1_dn[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[3].set_title("q_FD1_dn")
    fig.tight_layout()

    im = ax[4].imshow(q_DA_up)
    ax[4].set_xticks(np.arange(len(x)), labels=x)
    ax[4].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[4].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[4].text(j, i, q_DA_up[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[4].set_title("q_DA_up")
    fig.tight_layout()

    im = ax[5].imshow(q_DA_dn)
    ax[5].set_xticks(np.arange(len(x)), labels=x)
    ax[5].set_yticks(np.arange(len(Feature_Selection)), labels=Feature_Selection)
    plt.setp(ax[5].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(Feature_Selection)):
        for j in range(len(x)):
            text = ax[5].text(j, i, q_DA_dn[i, j].round(0),
                        ha="center", va="center", color="w")
    ax[5].set_title("q_DA_dn")
    fig.tight_layout()

    if save == True:
        plt.savefig('Result_plots/coefficients_heatmap.png', bbox_inches='tight')

    # plt.show()
import numpy as np

def Initialize_Activation_Array(Resolution,columnbased):

    # Initialize Activation array:

    Header = np.array([["Date"]])
    Time = np.arange(1, (Resolution+1) ).reshape(Resolution,1)

    Header2 = np.array([["Values"]])

    if columnbased == False: # Then the csv file will not be a concatenation of columns so one columns for values are created.
        FFR_comb = np.concatenate((Header, Header2),axis=1,dtype=object)
        FCR_N_up_comb = np.concatenate((Header, Header2),axis=1,dtype=object)
        FCR_N_dn_comb = np.concatenate((Header, Header2),axis=1,dtype=object)
        FCR_D_up_comb = np.concatenate((Header, Header2),axis=1,dtype=object)
        FCR_D_dn_comb = np.concatenate((Header, Header2),axis=1,dtype=object)

    else:
        FFR_comb = np.concatenate((Header, Time),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)
        FCR_N_up_comb = np.concatenate((Header, Time),axis=0,dtype=object)
        FCR_N_dn_comb = np.concatenate((Header, Time),axis=0,dtype=object)
        FCR_D_up_comb = np.concatenate((Header, Time),axis=0,dtype=object)
        FCR_D_dn_comb = np.concatenate((Header, Time),axis=0,dtype=object)


    return FFR_comb,FCR_N_up_comb,FCR_N_dn_comb,FCR_D_up_comb,FCR_D_dn_comb


# Function that take the frequency array of values between 0s and 1s and take the mean for the given resolution.
# Array = array of 0s and 1s that correspond to the activation
def Transform_to_Resolution(Array, Resolution, Default_number_of_data_points = 10*60*60*24):
    Array_Res_base = np.zeros((Resolution)) # Array of the size corresponding to the 

    #Missing_Data_Points = Default_number_of_data_points - np.size(Array)
    Data_Points_For_One_Resolution = int(Default_number_of_data_points/Resolution) #  ms -> h, 10 ms/s* 60 min/s * 60 h/min * 1 1/h

    for i in range(0,Resolution):
        idx1 = i*Data_Points_For_One_Resolution
        if i == Resolution:
            idx2 = np.size(Array)
        else:
            idx2 = (i+1)*Data_Points_For_One_Resolution

        Array_Res_base[i] = np.average(Array[idx1:idx2])
        #print('Index: ', idx1 ,':', idx2)
        #print('average: ', Array_hour_base[i])

    return Array_Res_base

def FFR_manipulation(FFR_arr):
    
    # Input:
    #FFR_arr. 0 and 1 for instances where the frequency has been below 49.7 Hz
    
    N = 10*60*60*24 # Total number of datapoints in dataset
    duration_Constant = 10*10 # Minimum Delivery time. 10 seconds
    duration_Linear_Decrease = 10*10 + 10 * 60 * 15
  
    # Find the indices where the value of FFR_arr are 1
    activation_indices = np.where(FFR_arr == 1)[0] # Setting 0 such that it does not take the whole array but only what is inside

    # Create the FFR activation array
    FFR_act = np.zeros(N)

    # Loop through the activation indices
    for i in activation_indices:

        # Set the corresponding datapoint in array2 to 1
        FFR_act[i] = 1
        end_idx1 = min(N,i+duration_Constant)
        # Use slicing and masking to set the next duration datapoints in array2 to 1
        FFR_act[i+1:end_idx1] = 1

        # Calculate the linear decrease
        x = np.linspace(1, 0, duration_Linear_Decrease - duration_Constant)

        # Use slicing and masking to set the next duration datapoints in array2 to the linear decrease
        end_idx2 = min(N,i+duration_Linear_Decrease)
        FFR_act[i+duration_Constant:end_idx2] = x[:end_idx2-i-duration_Constant]

    return FFR_act

# Transform all frequency csv files in a folder to one transformed file
# That follow FFR
def Transform_From_Freq_To_FFR_Act(df_arr, Resolution):

    # Frequency 
    Freq = df_arr[:,1]

    # Do the transformation in accordance to FFR rules
    FFR_act = np.zeros(Freq.size)

    # FFR
    FFR_freq_on_off = np.where((Freq <= 49.7), 1, 0)# Just an array for under 49.7. Will be used instead of seperation of 49.7, 49.6 and 49.5. So just everything below 49.7
    FFR_act = FFR_manipulation(FFR_freq_on_off) # Transform it from when the frequency occur to the actual activation

    # After the frequency has been transformed into values from 0 to 1 for each ms. Then transform it an equivalent resolution. 
    FFR_act_r = Transform_to_Resolution(FFR_act,Resolution)

    return FFR_act_r

# Transform all frequency csv files in a folder to one transformed file
# That follow FCR-D. With Generator convention.S
def Transform_From_Freq_To_FCR_D_Act(df_arr, Resolution):

    # Frequency 
    Freq = df_arr[:,1]

    # Do the transformation in accordance to FFR and FCR
    FCR_D_act_up = np.zeros(Freq.size)
    FCR_D_act_dn = np.zeros(Freq.size)

    # FCR-D
    #a = (1-0)/(49.5-49.9) ; b = 1 - a*49.5
    FCR_D_act_up = np.where((Freq <= 49.9) & (Freq > 49.5), (Freq-49.9)/-0.4, np.where(Freq <= 49.5,1,0))

    #a = (-1-0)/(50.5-50.1) ; b = -1 - a*50.5
    FCR_D_act_dn = np.where((Freq <= 50.5) & (Freq >= 50.1), (Freq-50.1)/0.4, np.where(Freq >= 50.5,1,0))

    # After the frequency has been transformed into values from 0 to 1 for each ms. Then transform it an equivalent resolution. 
    FCR_D_act_up_r = Transform_to_Resolution(FCR_D_act_up,Resolution)
    FCR_D_act_dn_r = Transform_to_Resolution(FCR_D_act_dn,Resolution)

    return FCR_D_act_up_r,FCR_D_act_dn_r


# Transform all frequency csv files in a folder to one transformed file
# That follow FCR-N
def Transform_From_Freq_To_FCR_N_Act(df_arr, Resolution):

    # Frequency 
    Freq = df_arr[:,1]

    # Do the transformation in accordance to FCR-N
    FCR_N_act_dn = np.zeros(Freq.size)
    FCR_N_act_up = np.zeros(Freq.size)

    # FCR-N
    # x1 = (49.9,1) , x2 = (50,0)
    a = (1-0)/(49.9-50)
    b = 1 - a*49.9
    FCR_N_act_up = np.where((Freq < 50) & (Freq >= 49.9), a*Freq + b, np.where(Freq <= 49.9,1,0))

    # x1 = (50.1,-1) , x2 = (50,0)
    a = (-1-0)/(50.1-50)
    b = -1 - a*50.1
    FCR_N_act_dn = np.where((Freq <= 50.1) & (Freq >= 50), a*Freq + b, np.where(Freq <= 50.1,1,0))


    # Exemption?
    # Every 15th minute is set to 0?
    

    # After the frequency has been transformed into values from 0 to 1 for each ms. Then transform it an equivalent resolution. 
    FCR_N_act_up_r = Transform_to_Resolution(FCR_N_act_up,Resolution)
    FCR_N_act_dn_r = Transform_to_Resolution(FCR_N_act_dn,Resolution)

    return FCR_N_act_up_r,FCR_N_act_dn_r






############################################################################################
########################## FREQUENCY EXAMINATION FUNCTIONS #################################
############################################################################################



# Function that takes the resolution wanted and provide the
# Inputs
# Resolution, the specific resolution
# Array, which the resolution index 
def Resolution_Index(Resolution, Array):
    Res = Resolution # The Resolution
    Res_idx = np.arange(start=0, stop=np.size(Array), step = Res)
    #Res_idx[-1] = len(Array) - 1 # Append the remaining if Resolution/step does not match exactly with the total amount 

    return Res_idx
    

# Function that takes the average, min, max and difference between min/max for a ranges of an array fitting to specific resolution
# Input: 
# Array. For example the frequency
# Resolution Array, an array with all the different resolution which one wants to inspect
# Empty_Array_data
def Construct_Resolution_Modified_Array_Data(Array,Resolution_idx,Empty_Array_data):
    #idx1 = 0 ; idx2 = 0 # Initialize the indeces
    Array_data = Empty_Array_data

    split_arrays = np.split(Array, Resolution_idx[1:-1]) # Include all the way to the end
    #split_arrays = np.split(Array, Resolution_idx[1::]) # Include only until the last index from Resolution_idx (so omit the remaining to the length of the array)
    #print(np.shape(split_arrays))
    #print(np.shape(split_arrays[1]))

    # initialize empty arrays to store the results
    split_min = np.empty(len(split_arrays))
    split_max = np.empty(len(split_arrays))
    split_avg = np.empty(len(split_arrays))
    split_std = np.empty(len(split_arrays))

    # calculate the minimum, maximum, and average of each split array
    for i, arr in enumerate(split_arrays):
        split_min[i] = np.min(arr)
        split_max[i] = np.max(arr)
        split_avg[i] = np.mean(arr)
        split_std[i] = np.std(arr)

    split_diff = split_max - split_min
    

    Array_data[:,0] = Resolution_idx[1::]
    Array_data[:,1] = split_min
    Array_data[:,2] = split_max
    Array_data[:,3] = split_avg
    Array_data[:,4] = split_std
    Array_data[:,5] = split_diff



    return Array_data



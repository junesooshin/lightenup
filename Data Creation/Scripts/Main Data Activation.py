# Data main file
import numpy as np
import pandas as pd

from TemporaryProcessing import Zip_File_To_Temporary
from TemporaryProcessing import Delete_Files_In_Temporary
from ArrayCreation import FileName_In_Folder
from ArrayCreation import open_File_And_Convert_To_Numpy_Array

import Create_FrequencyData 
import ArrayCreation 

# Frequency processing
#C:\Users\Daniel  BV\Desktop\Load flexibility\Data\Raw Data\Frequency

# Frequency data is in zip files seperated in months May, June, July, August and September. 
# In each zip files are there csv files for all the days in the month
# The below function opens the zip file in a temporary folder and process each csv file.
# By converting from frequencies to activation value between 0 and 1

# All of this should have its own frequency function

# The folders to open
InputFolderYear = np.array(["DK2 2022"]) # The folder seperating the year
InputFolderMonth = np.array(["-01.zip","-02.zip","-03.zip","-04.zip","-05.7z","-06.7z","-07.7z","-08.7z","-09.7z","-10.7z","-11.7z","-12.7z"]) # The folder seperating the month in the year
#InputFolderMonth = np.array(["-01.7z","-02.7z","-03.7z"]) # The folder seperating the month in the year

Frequency_Raw_Input_Path = 'C:\\Users\\Daniel  BV\\Desktop\\Thesis\\Code\\Data\\Raw Data\\Frequency' 
Frequency_Output_Path = 'C:\\Users\\Daniel  BV\\Desktop\\Thesis\\Code\\Data\\Processed Data\\Frequency\\Activation - No Scenarios' # Output path for frequency data

Temp_Path = 'C:\\Users\\Daniel  BV\\Desktop\\Thesis\\Code\\Data\\Temporary File' # Temporary file

create_FFR = False
create_FCR_D = True
create_FCR_N = False
# state a resolution [h, min, s]
Resolutions = np.array([24, 24*60, 24*60*60])
Resolution = Resolutions[0]

# Row based or column based output file. Row based: (n*24,1) Column based: (24, n)
columnbased = True

# Initialize the activation array:
FFR_comb,FCR_N_up_comb,FCR_N_dn_comb,FCR_D_up_comb,FCR_D_dn_comb = Create_FrequencyData.Initialize_Activation_Array(Resolution,columnbased)


for year in InputFolderYear:
    for month in InputFolderMonth:

        # Open zip file folder and extract it to temporary file
        filename = year[-4:] + month #Create the fileName
        InputFolder = Frequency_Raw_Input_Path + "/" + year 
        Zip_File_To_Temporary(filename, InputFolder, Temp_Path) # Zip to the temporary folder



        # Count all csv files in temporary folder
        FileNames = FileName_In_Folder(Temp_Path) # get names of all files in folder
        Number_of_Files = np.size(FileNames[:]) # number of file in folder

        # For each file do as follow
        for f in range(0,Number_of_Files):
            
            # Pick the file, open it and convert to a numpy array
            file = FileNames[f]
            df_arr = open_File_And_Convert_To_Numpy_Array(Temp_Path, file)

            # Give the proper shape and a header (the date)
            Header = np.array([[file[:-4]]])

            # Create the time index for column based
            input_date = pd.to_datetime(Header[0][0]) # Convert the input string to a pandas datetime object
            end_date = input_date + pd.offsets.Hour(23) # Add one more day to be the end day..
            time_index = pd.date_range(start=input_date,end=end_date, periods=Resolution) # Generate a time index with the specified resolution
            time_index = pd.DataFrame(time_index).astype(str).values

            if create_FFR == True:
    
                FFR_act_r = Create_FrequencyData.Transform_From_Freq_To_FFR_Act(df_arr,Resolution) # Convert from frequency and ms to activation and hour 

                if columnbased == True:
                    FFR_act_r_upd = np.concatenate((Header, FFR_act_r.reshape(Resolution,1)),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)

                    FFR_comb = np.concatenate((FFR_comb,FFR_act_r_upd),axis=1,dtype=object)# Append the previous files with the new updated ones for the opened CSV file
                else:

                    FFR_act_r_upd = np.hstack((time_index.reshape(-1,1),FFR_act_r.reshape(Resolution,1)))
                    FFR_comb = np.concatenate((FFR_comb,FFR_act_r_upd),axis=0,dtype=object)# Append the previous files with the new updated ones for the opened CSV file



            elif create_FCR_N == True:

                FCR_N_act_up_r,FCR_N_act_dn_r = Create_FrequencyData.Transform_From_Freq_To_FCR_N_Act(df_arr,Resolution) # Convert from frequency and ms to activation and hour 


                if columnbased == True:
                    FCR_N_act_up_r_upd = np.concatenate((Header, FCR_N_act_up_r.reshape(Resolution,1)),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)
                    FCR_N_act_dn_r_upd = np.concatenate((Header, FCR_N_act_dn_r.reshape(Resolution,1)),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)
                    
                    FCR_N_up_comb = np.concatenate((FCR_N_up_comb,FCR_N_act_up_r_upd),axis=1,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                    FCR_N_dn_comb = np.concatenate((FCR_N_dn_comb,FCR_N_act_dn_r_upd),axis=1,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                else:
                    FCR_N_act_up_r_upd = np.concatenate((time_index, FCR_N_act_up_r.reshape(Resolution,1)),axis=1,dtype=object) # Append Header on top and Time underneath, making it a (25,1)
                    FCR_N_act_dn_r_upd = np.concatenate((time_index, FCR_N_act_dn_r.reshape(Resolution,1)),axis=1,dtype=object) # Append Header on top and Time underneath, making it a (25,1)

                    FCR_N_up_comb = np.concatenate((FCR_N_up_comb,FCR_N_act_up_r_upd),axis=0,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                    FCR_N_dn_comb = np.concatenate((FCR_N_dn_comb,FCR_N_act_dn_r_upd),axis=0,dtype=object) # Append the previous files with the new updated ones for the opened CSV file


            elif create_FCR_D == True:

                FCR_D_act_up_r,FCR_D_act_dn_r = Create_FrequencyData.Transform_From_Freq_To_FCR_D_Act(df_arr,Resolution) # Convert from frequency and ms to activation and hour 
                #print(np.shape(FCR_D_act_up_r))
                #print(np.shape(time_index))
                #print(np.shape(FCR_D_act_up_r.reshape(-1,1)))
                #print(np.shape(time_index.reshape(-1,1))) #Fails here why?
                if columnbased== True:
                    #np.hstack((time_index.reshape(-1,1),FFR_act_r.reshape(Resolution,1)))
                    FCR_D_act_up_r_upd = np.concatenate((Header, FCR_D_act_up_r.reshape(Resolution,1)),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)
                    FCR_D_act_dn_r_upd = np.concatenate((Header, FCR_D_act_dn_r.reshape(Resolution,1)),axis=0,dtype=object) # Append Header on top and Time underneath, making it a (25,1)

                    FCR_D_up_comb = np.concatenate((FCR_D_up_comb,FCR_D_act_up_r_upd),axis=1,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                    FCR_D_dn_comb = np.concatenate((FCR_D_dn_comb,FCR_D_act_dn_r_upd),axis=1,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                else: 
                    FCR_D_act_up_r_upd = np.hstack( (time_index.reshape(-1,1) , FCR_D_act_up_r.reshape(Resolution,1))) # Append Header on top and Time underneath, making it a (25,1)
                    FCR_D_act_dn_r_upd = np.hstack((time_index.reshape(-1,1), FCR_D_act_dn_r.reshape(Resolution,1))) # Append Header on top and Time underneath, making it a (25,1)

                    FCR_D_up_comb = np.concatenate((FCR_D_up_comb,FCR_D_act_up_r_upd),axis=0,dtype=object) # Append the previous files with the new updated ones for the opened CSV file
                    FCR_D_dn_comb = np.concatenate((FCR_D_dn_comb,FCR_D_act_dn_r_upd),axis=0,dtype=object) # Append the previous files with the new updated ones for the opened CSV file

            print('File ', file, ' processed')

        # Delete all data in the temporary folder
        Delete_Files_In_Temporary(Temp_Path)



if create_FFR == True:
    DF_FFR = pd.DataFrame(FFR_comb)
    DF_FFR.to_csv(Frequency_Output_Path + "/FFR_act.csv",index=False,header=False)

elif create_FCR_D == True:
    DF_FCR_D_up = pd.DataFrame(FCR_D_up_comb)
    DF_FCR_D_dn = pd.DataFrame(FCR_D_dn_comb)
    DF_FCR_D_up.to_csv(Frequency_Output_Path + "/FCR_D_act_up.csv",index=False,header=False)
    DF_FCR_D_dn.to_csv(Frequency_Output_Path + "/FCR_D_act_dn.csv",index=False,header=False)

elif create_FCR_N == True:
    DF_FCR_N_up = pd.DataFrame(FCR_N_up_comb)
    DF_FCR_N_dn = pd.DataFrame(FCR_N_dn_comb)
    DF_FCR_N_up.to_csv(Frequency_Output_Path + "/FCR_N_act_up.csv",index=False,header=False)
    DF_FCR_N_dn.to_csv(Frequency_Output_Path + "/FCR_N_act_dn.csv",index=False,header=False)








  




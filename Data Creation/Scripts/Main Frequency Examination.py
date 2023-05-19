

# Data main file
import numpy as np
import pandas as pd
from TemporaryProcessing import Zip_File_To_Temporary
from TemporaryProcessing import Delete_Files_In_Temporary
from ArrayCreation import FileName_In_Folder
from ArrayCreation import open_File_And_Convert_To_Numpy_Array

from Create_FrequencyData import Resolution_Index
from Create_FrequencyData import Construct_Resolution_Modified_Array_Data

import ArrayCreation 
import time
# Frequency processing
#C:\Users\Daniel  BV\Desktop\Load flexibility\Data\Raw Data\Frequency

# Frequency data is in zip files seperated in months May, June, July, August and September. 
# In each zip files are there csv files for all the days in the month
# The below function opens the zip file in a temporary folder and process each csv file.
# By converting from frequencies to activation value between 0 and 1

# All of this should have its own frequency function

# The folders to open
InputFolderYear = np.array(["DK2 2022","DK2 2023"]) # The folder seperating the year

InputFolderMonth = dict({"DK2 2022": ["-01.zip","-02.zip","-03.zip","-04.zip","-05.7z","-06.7z","-07.7z","-08.7z","-09.7z","-10.7z","-11.7z","-12.7z"],
                        "DK2 2023": ["-01.7z","-02.7z","-03.7z"]}) # The folder seperating the month in the year

Frequency_Raw_Input_Path = 'C:/Users/Daniel  BV/Desktop/Thesis/Code/Data/Raw Data/Frequency' 
Frequency_Output_Path = 'C:/Users/Daniel  BV/Desktop/Thesis/Code/Data/Processed Data/Frequency/Frequency Examination' # Output path for frequency data

Temp_Path = 'C:/Users/Daniel  BV/Desktop/Thesis/Code/Data/Temporary File' # Temporary file

Array_data = np.empty(shape=[0, 6])

start_time_1 = time.time() # Record time


for year in InputFolderYear:
    for month in InputFolderMonth[year]:

        # Open zip file folder and extract it to temporary file
        filename = year[-4:] + month #Create the fileName
        InputFolder = Frequency_Raw_Input_Path + "/" + year 
        Zip_File_To_Temporary(filename, InputFolder, Temp_Path) # Zip to the temporary folder



        # Count all csv files in temporary folder
        FileNames = FileName_In_Folder(Temp_Path) # get names of all files in folder
        Number_of_Files = np.size(FileNames[:]) # number of file in folder

        # Computiational time?
        # For each file do as follow
        for f in range(0,Number_of_Files):
            
            # Pick the file, open it and convert to a numpy array
            file = FileNames[f]
            df_arr = open_File_And_Convert_To_Numpy_Array(Temp_Path, file)
    
           
            Freq = df_arr[:,1] # Frequency
            #print(np.size(Freq))
            Resolution_idx = Resolution_Index(10, Freq) # Specify the Resolution indeces
            Empty_Array_data = np.zeros(( (np.size(Resolution_idx)-1) , 6 ) ) # Initiate an array of size (Total array size/ Resolution) and 4 columns ( Min, Max, Avg, Max-Min )
        
            temp_Array_data = Construct_Resolution_Modified_Array_Data(Freq, Resolution_idx,Empty_Array_data) # Populate the array
            # Shape of the array:
            #print(np.shape(temp_Array_data))

            
            # Append to row on rows
            Array_data = np.concatenate((Array_data, temp_Array_data),axis=0) # Append
            print(np.shape(Array_data)) 
            
            print('File ', file, ' processed')


        # Delete all data in the temporary folder
        Delete_Files_In_Temporary(Temp_Path)

# Does not work yet
DF_Array_data = pd.DataFrame(Array_data)

DF_Array_data.to_csv(Frequency_Output_Path + "/DF_Array_data.csv",index=False,header=False)

end_time_1 = time.time()
print('Done after ' , (end_time_1 - start_time_1)/60 , ' minutes')





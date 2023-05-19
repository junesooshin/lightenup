
import numpy as np
import os
import pandas as pd


def Load_Numpy_Array(InputPath, ArrayName): # Load numpy files

    Fullpath = InputPath + '/' + ArrayName # The full path

    # Load array in folder
    return np.load(Fullpath, allow_pickle=True)

def Save_Numpy_Array_To_Path(Array, OutputPath, SavedArrayName): # Save numpy array as a npy file

    Fullpath = OutputPath + '/' + SavedArrayName + '.npy' # Full path

    with open(Fullpath, 'wb') as f:

        return np.save(f, Array )



def open_File_And_Convert_To_Numpy_Array(path, FileName, FileType = 'csv'): # Converting files to numpy by use of pandas
    path_pd = path
    FileName_pd = FileName
    Filetype_pd = FileType
    
    Full_path = path_pd + '/' + FileName_pd # Full Path

    if FileType == 'csv': #Open csv files
        df = pd.read_csv(Full_path)
        df_arr = df.to_numpy()
        
    return df_arr # Data in numpy array format


def Count_Files_In_Folder(dir_path):
    
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1

    return count

def FileName_In_Folder(dir_path):

    path = dir_path # Define path
    Number_of_files = Count_Files_In_Folder(path) # Find the number of files in path

    FileNames = np.empty([Number_of_files],dtype = object) # Initialize Array

    count = 0
    for item in os.listdir(dir_path):
        
        FileNames[count] = item #Insert the item name to the array FileNames
        count += 1

    return FileNames # output all the fileNames from the given folder


def Create_CSV_file_From_Array(count,Array,  Output_path,CSV_File_Name,Header1='Time',Header2='Freq_hour_val'):

    if count == 0: # if it is the first then make the headers
        # Initiate the file by making the Header
        header = np.array([[Header1],[Header2]])

        rows = ["{},{}".format(header[0,0],header[1,0])] 
        text = "\n".join(rows)
        
        with open(Output_path + '/' + CSV_File_Name, 'w') as f:
            f.write(text)

    # Write the rows and columns from array  to a text format
    rows = ["{},{}".format(i, j) for i, j in Array]
    text = "\n".join(rows) # Insert a newline between each row element
    text = "\n" + text

    with open(Output_path + '/' + CSV_File_Name ,'a') as f: # Append content
        f.write(text)



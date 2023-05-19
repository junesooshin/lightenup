
# This sub script is for all the handling of the temporary folder. It includes functions that handle:

#   - Extraction of zip files from data source to temporary folder
#   - Deletion of all data in temporary folder

# Libraries utilized:
import os
import numpy as np
from zipfile import ZipFile 
import py7zr

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


dir_path = r'C:/Users/Daniel  BV/Desktop/Load flexibility/Data/Frequency/Zip data'
zipped_path = 'C:/Users/Daniel  BV/Desktop/Load flexibility/Data/Frequency/DK2 2021'




def Zip_File_To_Temporary(file_name, Input_path, Output_path):
    if 'zip' in file_name:
        Zip_File_path =  Input_path + '\\' + file_name 
        # open the zip file in read mode
        with ZipFile(Zip_File_path, 'r') as zip: 

            # extract all files to another directory
            zip.extractall(Output_path)
            print(file_name , ' was extracted!')
    elif '7z' in file_name:

        Zip_File_path =  Input_path + '\\' + file_name 
        with py7zr.SevenZipFile(Zip_File_path, mode='r') as z:
            z.extractall(Output_path)
            print(file_name , ' was extracted!')

    return



def Delete_Files_In_Temporary(Temp_path): # Function that delete files in the temporary folder

    for f in os.listdir(Temp_path):
        os.remove(os.path.join(Temp_path, f))

    return


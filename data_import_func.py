import numpy as np
import pandas as pd
import requests
import json 

def import_data_API(start_date, end_date, market):
    """
    Function to import data from API
    Input: start_date, end_date, market
    Output: DataFrame
    """
    limit = "1000000" # Set limit

    if 'spot' in market:
        column = "HourDK, PriceArea, SpotPriceEUR"
        url = 'https://api.energidataservice.dk/dataset/Elspotprices?filter={"PriceArea":["DK2"]}'
    elif 'bal' in market:
        column = "HourDK, PriceArea, BalancingPowerPriceUpEUR, BalancingPowerPriceDownEUR"
        url='https://api.energidataservice.dk/dataset/RegulatingBalancePowerdata?filter={"PriceArea":["DK2"]}'
    elif 'FCRD' in market:
        column = "HourDK, PriceArea, ProductName, AuctionType, PurchasedVolumeLocal, PurchasedVolumeTotal, PriceTotalEUR"
        url='https://api.energidataservice.dk/dataset/FcrNdDK2'
        if 'up' in market:
            if 'D-1' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D upp"),"AuctionType": "D-1"}
            elif 'D-2' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D upp"),"AuctionType": "D-2"}
            elif 'total' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D upp"),"AuctionType": "Total"}
            else:
                print("mention D-1 or D-2!")
        elif 'down' in market:
            if 'D-1' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D ned"),"AuctionType": "D-1"} 
            elif 'D-2' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D ned"),"AuctionType": "D-2"}
            elif 'total' in market:
                filter_dict = {"PriceArea": "DK2","ProductName": ("FCR-D ned"),"AuctionType": "Total"}
            else:
                print("mention D-1 or D-2!")
        else:
            print("mention up or down!")
        filter_json = json.dumps(filter_dict)
    else:
        print("wrong market input!")   
    
    parameters = {
        "start": start_date,
        "end": end_date,
        "columns": column,
        "limit": limit
    }
    if 'FCRD' in market:
        parameters["filter"] = filter_json
        
    response = requests.get(url=url, params=parameters)
    result = response.json()
    records = result['records']
    df = pd.DataFrame(records)
  
    return df

def fix_indexing(df, start_date, end_date):
    """
    Function to remove duplicates, fill NaN to missing timestep
    Input: DataFrame, start_date, end_date
    Output: DataFrame
    """
    df_new = df.copy()
    correct_index = pd.date_range(start=start_date, end=end_date, inclusive="left", freq='H')
    df_new.drop_duplicates(subset='HourDK', keep='first',inplace=True)
    df_new.index = pd.to_datetime(df_new['HourDK'])
    df_new = df_new.reindex(correct_index, fill_value = np.nan)

    return df_new

def NaN_FCR_D(df_D_1_Down,df_D_2_Down,df_Total_Down,df_D_1_Up,df_D_2_Up,df_Total_Up,printstatement=True):
    """
    There exist inconsistencies between D-1, D-2 and Total. Before doing anything with changing NaN values to 0 or interpolation align D-1, D-2 and Total

    Parameters:
    df_D_1_Down (pandas.DataFrame): The dataframe for D-1 downreg, 4 columns..
    df_D_2_Down (pandas.DataFrame): The dataframe for D-2 downreg..
    df_Total_Down (pandas.DataFrame): The dataframe for Total downreg..

    Returns:
    pandas.df_D_1_Down_mod: The modified dataframe for D-1 downreg
    """
    #print(df_D_1_Down.loc[0,'HourDK'])
    #Correct_Index_Dn = pd.date_range(start=df_D_1_Down.loc[0,'HourDK'], end=df_D_1_Down.loc[-1,'HourDK'], freq='H') # Set the correct index such that all time instances are present
    #Correct_Index_Up = pd.date_range(start=df_D_1_Up.loc[0,'HourDK'], end=df_D_1_Up.loc[-1,'HourDK'], freq='H') # Set the correct index such that all time instances are present
    
    common_index_Down = df_D_1_Down.index.intersection(df_D_2_Down.index).intersection(df_Total_Down.index)
    common_index_Up = df_D_1_Up.index.intersection(df_D_2_Up.index).intersection(df_Total_Up.index)
    #print(common_index_Down.shape) 
    
    # Define all the nan_masks
    nan_mask_D_1_Down = df_D_1_Down["PriceTotalEUR"].isna()
    nan_mask_D_2_Down = df_D_2_Down["PriceTotalEUR"].isna()
    nan_mask_Total_Down = df_Total_Down["PriceTotalEUR"].isna()

    nan_mask_D_1_Down_V = df_D_1_Down["PurchasedVolumeTotal"].isna()
    nan_mask_D_2_Down_V = df_D_2_Down["PurchasedVolumeTotal"].isna()
    nan_mask_Total_Down_V = df_Total_Down["PurchasedVolumeTotal"].isna()

    nan_mask_D_1_Up = df_D_1_Up["PriceTotalEUR"].isna()
    nan_mask_D_2_Up = df_D_2_Up["PriceTotalEUR"].isna()
    nan_mask_Total_Up = df_Total_Up["PriceTotalEUR"].isna()

    nan_mask_D_1_Up_V = df_D_1_Up["PurchasedVolumeTotal"].isna()
    nan_mask_D_2_Up_V = df_D_2_Up["PurchasedVolumeTotal"].isna()
    nan_mask_Total_Up_V = df_Total_Up["PurchasedVolumeTotal"].isna()

    #### Cases where there are no prices in D-1, D-2 and Total
    result_index_No_Data_Dn = common_index_Down[(nan_mask_D_1_Down.loc[common_index_Down] & 
                            nan_mask_D_2_Down.loc[common_index_Down] & 
                            nan_mask_Total_Down.loc[common_index_Down])]

    result_index_No_Data_Dn_V = common_index_Down[(nan_mask_D_1_Down_V.loc[common_index_Down] & 
                                nan_mask_D_2_Down_V.loc[common_index_Down] & 
                                nan_mask_Total_Down_V.loc[common_index_Down])]

    result_index_No_Data_Up = common_index_Up[(nan_mask_D_1_Up.loc[common_index_Up] & 
                                nan_mask_D_2_Up.loc[common_index_Up] & 
                                nan_mask_Total_Up.loc[common_index_Up])]

    result_index_No_Data_Up_V = common_index_Up[(nan_mask_D_1_Up_V.loc[common_index_Up] & 
                                nan_mask_D_2_Up_V.loc[common_index_Up] & 
                                nan_mask_Total_Up_V.loc[common_index_Up])]

    # Cases where there is no D-1 price, but D-2 and Total
    result_index_no_D_1_price_Down = common_index_Down[( (nan_mask_D_1_Down.loc[common_index_Down]==True) & 
                            ((nan_mask_D_2_Down.loc[common_index_Down]==False) & 
                            (nan_mask_Total_Down.loc[common_index_Down]==False)) )]
    
    result_index_no_D_1_Volume_Down = common_index_Down[( (nan_mask_D_1_Down_V.loc[common_index_Down]==True) & 
                                ((nan_mask_D_2_Down_V.loc[common_index_Down]==False) &
                                (nan_mask_Total_Down_V.loc[common_index_Down]==False)) )]
    
    result_index_no_D_1_prices_Up = common_index_Up[( (nan_mask_D_1_Up.loc[common_index_Up]==True) & 
                            ((nan_mask_D_2_Up.loc[common_index_Up]==False) & 
                            (nan_mask_Total_Up.loc[common_index_Up]==False)) )]
    
    result_index_no_D_1_Volume_Up = common_index_Up[( (nan_mask_D_1_Up_V.loc[common_index_Up]==True) & 
                                ((nan_mask_D_2_Up_V.loc[common_index_Up]==False) & 
                                (nan_mask_Total_Up_V.loc[common_index_Up]==False)) )]

    # Cases where there are price in D-1 and Total but not in D-2
    result_index_no_D_2_price_Down = common_index_Down[( (nan_mask_D_1_Down.loc[common_index_Down]==False) & 
                            (nan_mask_D_2_Down.loc[common_index_Down]==True) & 
                            (nan_mask_Total_Down.loc[common_index_Down]==False) )]
    result_index_no_D_2_Volume_Down = common_index_Down[( (nan_mask_D_1_Down_V.loc[common_index_Down]==False) & 
                                (nan_mask_D_2_Down_V.loc[common_index_Down]==True) &  
                                (nan_mask_Total_Down_V.loc[common_index_Down]==False) )]
    
    result_index_no_D_2_prices_Up = common_index_Up[( (nan_mask_D_1_Up.loc[common_index_Up]==False) & 
                            (nan_mask_D_2_Up.loc[common_index_Up]==True) &
                            (nan_mask_Total_Up.loc[common_index_Up]==False) )]
    result_index_no_D_2_Volume_Up = common_index_Up[( (nan_mask_D_1_Up_V.loc[common_index_Up]==False) & 
                                (nan_mask_D_2_Up_V.loc[common_index_Up]==True) & 
                                (nan_mask_Total_Up_V.loc[common_index_Up]==False) )]

    # Cases with no D-1 or D-2
    result_index_no_D_1_D_2_price_Down = common_index_Down[( (nan_mask_D_1_Down.loc[common_index_Down]==True) & 
                            ((nan_mask_D_2_Down.loc[common_index_Down]==True) & 
                            (nan_mask_Total_Down.loc[common_index_Down]==False)) )]
    result_index_no_D_1_D_2_Volume_Down = common_index_Down[( (nan_mask_D_1_Down_V.loc[common_index_Down]==True) & 
                                ((nan_mask_D_2_Down_V.loc[common_index_Down]==True) & 
                                (nan_mask_Total_Down_V.loc[common_index_Down]==False)) )]

    result_index_no_D_1_D_2_prices_Up = common_index_Up[( (nan_mask_D_1_Up.loc[common_index_Up]==True) & 
                                ((nan_mask_D_2_Up.loc[common_index_Up]==True) & 
                                (nan_mask_Total_Up.loc[common_index_Up]==False)) )]
    result_index_no_D_1_D_2_Volume_Up = common_index_Up[( (nan_mask_D_1_Up_V.loc[common_index_Up]==True) & 
                                ((nan_mask_D_2_Up_V.loc[common_index_Up]==True) & 
                                (nan_mask_Total_Up_V.loc[common_index_Up]==False)) )]
    df_D_1_Down_mod = df_D_1_Down
    df_D_1_Up_mod = df_D_1_Up
    df_D_2_Down_mod = df_D_2_Down
    df_D_2_Up_mod = df_D_2_Up

    #### HANDLE THE CASES WHERE TOTAL, D-2 AND D-1 ARE NOT ALIGNED!!
    ################ NO D-1 VALUES ###################
    # When the volumes are the same between D-2 and Total then the price is the same
    SamePrice = result_index_no_D_1_price_Down[df_D_2_Down.loc[result_index_no_D_1_price_Down,'PurchasedVolumeTotal'] == df_Total_Down.loc[result_index_no_D_1_price_Down,'PurchasedVolumeTotal']]
    DifferentPrice = result_index_no_D_1_price_Down[df_D_2_Down.loc[result_index_no_D_1_price_Down,'PurchasedVolumeTotal'] != df_Total_Down.loc[result_index_no_D_1_price_Down,'PurchasedVolumeTotal']]
    
    df_D_1_Down_mod.loc[SamePrice,'PriceTotalEUR'] = 0
    df_D_1_Down_mod.loc[SamePrice,'PurchasedVolumeLocal'] = 0
    df_D_1_Down_mod.loc[SamePrice,'PurchasedVolumeTotal'] = 0

    # The ones with different prices the volume start by getting adjusted
    df_D_1_Down_mod.loc[DifferentPrice,'PurchasedVolumeTotal'] = df_Total_Down.loc[DifferentPrice,'PurchasedVolumeTotal'] - df_D_2_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_2 = df_D_2_Down.loc[DifferentPrice,'PriceTotalEUR'] * df_D_2_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_Total = df_Total_Down.loc[DifferentPrice,'PriceTotalEUR'] * df_Total_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_1 = p_Total - p_D_2
    df_D_1_Down_mod.loc[DifferentPrice,'PriceTotalEUR'] = p_D_1 / df_D_1_Down_mod.loc[DifferentPrice,'PurchasedVolumeTotal']
    df_D_1_Down_mod.loc[DifferentPrice,'PurchasedVolumeLocal'] = 0 # Set it still to 0 here.
    
    # Do the same for upregulation
    # SOMETHING WRONG HERE!!
    SamePrice = result_index_no_D_1_prices_Up[df_D_2_Up.loc[result_index_no_D_1_prices_Up,'PurchasedVolumeTotal'] == df_Total_Up.loc[result_index_no_D_1_prices_Up,'PurchasedVolumeTotal']]
    DifferentPrice = result_index_no_D_1_prices_Up[df_D_2_Up.loc[result_index_no_D_1_prices_Up,'PurchasedVolumeTotal'] != df_Total_Up.loc[result_index_no_D_1_prices_Up,'PurchasedVolumeTotal']]
    
    df_D_1_Up_mod.loc[SamePrice,'PriceTotalEUR'] = 0
    df_D_1_Up_mod.loc[SamePrice,'PurchasedVolumeLocal'] = 0
    df_D_1_Up_mod.loc[SamePrice,'PurchasedVolumeTotal'] = 0
    
    # The ones with different prices the volume start by getting adjusted
    df_D_1_Up_mod.loc[DifferentPrice,'PurchasedVolumeTotal'] = df_Total_Up.loc[DifferentPrice,'PurchasedVolumeTotal'] - df_D_2_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_2 = df_D_2_Up.loc[DifferentPrice,'PriceTotalEUR'] * df_D_2_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_Total = df_Total_Up.loc[DifferentPrice,'PriceTotalEUR'] * df_Total_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_1 = p_Total - p_D_2
    df_D_1_Up_mod.loc[DifferentPrice,'PriceTotalEUR'] = p_D_1 / df_D_1_Up_mod.loc[DifferentPrice,'PurchasedVolumeTotal']
    df_D_1_Up_mod.loc[DifferentPrice,'PurchasedVolumeLocal'] = 0 # Set it still to 0 here.

    ################ NO D-2 VALUES ###################
    # When the volumes are the same between D-1 and Total then the price is the same
    SamePrice = result_index_no_D_2_price_Down[df_D_1_Down.loc[result_index_no_D_2_price_Down,'PurchasedVolumeTotal'] == df_Total_Down.loc[result_index_no_D_2_price_Down,'PurchasedVolumeTotal']]
    DifferentPrice = result_index_no_D_2_price_Down[df_D_1_Down.loc[result_index_no_D_2_price_Down,'PurchasedVolumeTotal'] != df_Total_Down.loc[result_index_no_D_2_price_Down,'PurchasedVolumeTotal']]
    
    df_D_2_Down_mod.loc[SamePrice,'PriceTotalEUR'] = 0
    df_D_2_Down_mod.loc[SamePrice,'PurchasedVolumeLocal'] = 0
    df_D_2_Down_mod.loc[SamePrice,'PurchasedVolumeTotal'] = 0

    # The ones with different prices the volume start by getting adjusted
    df_D_2_Down_mod.loc[DifferentPrice,'PurchasedVolumeTotal'] = df_Total_Down.loc[DifferentPrice,'PurchasedVolumeTotal'] - df_D_1_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_1 = df_D_1_Down.loc[DifferentPrice,'PriceTotalEUR'] * df_D_1_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_Total = df_Total_Down.loc[DifferentPrice,'PriceTotalEUR'] * df_Total_Down.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_2 = p_Total - p_D_1
    df_D_2_Down_mod.loc[DifferentPrice,'PriceTotalEUR'] = p_D_2 / df_D_2_Down_mod.loc[DifferentPrice,'PurchasedVolumeTotal']
    df_D_2_Down_mod.loc[DifferentPrice,'PurchasedVolumeLocal'] = 0 # Set it still to 0 here.
    
    # Do the same for upregulation
    SamePrice = result_index_no_D_2_prices_Up[df_D_1_Up.loc[result_index_no_D_2_prices_Up,'PurchasedVolumeTotal'] == df_Total_Up.loc[result_index_no_D_2_prices_Up,'PurchasedVolumeTotal']]
    DifferentPrice = result_index_no_D_2_prices_Up[df_D_1_Up.loc[result_index_no_D_2_prices_Up,'PurchasedVolumeTotal'] != df_Total_Up.loc[result_index_no_D_2_prices_Up,'PurchasedVolumeTotal']]
    
    df_D_2_Up_mod.loc[SamePrice,'PriceTotalEUR'] = 0
    df_D_2_Up_mod.loc[SamePrice,'PurchasedVolumeLocal'] = 0
    df_D_2_Up_mod.loc[SamePrice,'PurchasedVolumeTotal'] = 0

    # The ones with different prices the volume start by getting adjusted
    df_D_2_Up_mod.loc[DifferentPrice,'PurchasedVolumeTotal'] = df_Total_Up.loc[DifferentPrice,'PurchasedVolumeTotal'] - df_D_1_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_1 = df_D_1_Up.loc[DifferentPrice,'PriceTotalEUR'] * df_D_1_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_Total = df_Total_Up.loc[DifferentPrice,'PriceTotalEUR'] * df_Total_Up.loc[DifferentPrice,'PurchasedVolumeTotal']
    p_D_2 = p_Total - p_D_1
    df_D_2_Up_mod.loc[DifferentPrice,'PriceTotalEUR'] = p_D_2 / df_D_2_Up_mod.loc[DifferentPrice,'PurchasedVolumeTotal']
    df_D_2_Up_mod.loc[DifferentPrice,'PurchasedVolumeLocal'] = 0 # Set it still to 0 here.

    ################ NO D-2 or D-1 VALUES ###################
    # Then total D-2 is just set the same as total
    
    df_D_1_Down_mod.loc[result_index_no_D_1_D_2_price_Down,'PriceTotalEUR'] = 0
    df_D_1_Down_mod.loc[result_index_no_D_1_D_2_Volume_Down,'PurchasedVolumeLocal'] = 0
    df_D_1_Down_mod.loc[result_index_no_D_1_D_2_Volume_Down,'PurchasedVolumeTotal'] = 0

    df_D_2_Down_mod.loc[result_index_no_D_1_D_2_price_Down,'PriceTotalEUR'] = df_Total_Down.loc[result_index_no_D_1_D_2_price_Down,'PriceTotalEUR']
    df_D_2_Down_mod.loc[result_index_no_D_1_D_2_Volume_Down,'PurchasedVolumeLocal'] = 0
    df_D_2_Down_mod.loc[result_index_no_D_1_D_2_Volume_Down,'PurchasedVolumeTotal'] = df_Total_Down.loc[result_index_no_D_1_D_2_Volume_Down,'PurchasedVolumeTotal']

    df_D_1_Up_mod.loc[result_index_no_D_1_D_2_prices_Up,'PriceTotalEUR'] = 0
    df_D_1_Up_mod.loc[result_index_no_D_1_D_2_Volume_Up,'PurchasedVolumeLocal'] = 0
    df_D_1_Up_mod.loc[result_index_no_D_1_D_2_Volume_Up,'PurchasedVolumeTotal'] = 0

    df_D_2_Up_mod.loc[result_index_no_D_1_D_2_prices_Up,'PriceTotalEUR'] = df_Total_Up.loc[result_index_no_D_1_D_2_prices_Up,'PriceTotalEUR']
    df_D_2_Up_mod.loc[result_index_no_D_1_D_2_Volume_Up,'PurchasedVolumeLocal'] = 0
    df_D_2_Up_mod.loc[result_index_no_D_1_D_2_Volume_Up,'PurchasedVolumeTotal'] = df_Total_Up.loc[result_index_no_D_1_D_2_Volume_Up,'PurchasedVolumeTotal']

    if printstatement ==True:
        # Total number of NaN
        print("Total NaN: nan_mask_D_1_Down", common_index_Down[nan_mask_D_1_Down.loc[common_index_Down]==True].shape)
        print("Total NaN: nan_mask_D_1_Down_V", common_index_Down[nan_mask_D_1_Down_V.loc[common_index_Down]==True].shape)
        print("Total NaN: nan_mask_D_2_Down", common_index_Down[nan_mask_D_2_Down.loc[common_index_Down]==True].shape)
        print("Total NaN: nan_mask_D_2_Down_V", common_index_Down[nan_mask_D_2_Down_V.loc[common_index_Down]==True].shape)
        print("Total NaN: nan_mask_Total_Down", common_index_Down[nan_mask_Total_Down.loc[common_index_Down]==True].shape)
        print("Total NaN: nan_mask_Total_Down_V", common_index_Down[nan_mask_Total_Down_V.loc[common_index_Down]==True].shape)
        print("")
        print("Total NaN: nan_mask_D_1_Up", common_index_Up[nan_mask_D_1_Up.loc[common_index_Up]==True].shape)
        print("Total NaN: nan_mask_D_1_Up_V", common_index_Up[nan_mask_D_1_Up_V.loc[common_index_Up]==True].shape)
        print("Total NaN: nan_mask_D_2_Up", common_index_Up[nan_mask_D_2_Up.loc[common_index_Up]==True].shape)
        print("Total NaN: nan_mask_D_2_Up_V", common_index_Up[nan_mask_D_2_Up_V.loc[common_index_Up]==True].shape)
        print("Total NaN: nan_mask_Total_Up", common_index_Up[nan_mask_Total_Up.loc[common_index_Up]==True].shape)
        print("Total NaN: nan_mask_Total_Up_V", common_index_Up[nan_mask_Total_Up_V.loc[common_index_Up]==True].shape)
        print("")
        print("No prices for down ",result_index_No_Data_Dn.shape)
        print("No volume for down ",result_index_No_Data_Dn_V.shape)
        print("No prices for up ",result_index_No_Data_Up.shape)
        print("No volume for up ",result_index_No_Data_Up_V.shape)
        print("")
        print("No D-1 prices for down ",result_index_no_D_1_price_Down.shape)
        print("No D-1 volume for down ",result_index_no_D_1_Volume_Down.shape)
        print("No D-1 prices for up ", result_index_no_D_1_prices_Up.shape)
        print("No D-1 volume for up ",result_index_no_D_1_Volume_Up.shape)
        print("")
        print("No D-2 prices for down ",result_index_no_D_2_price_Down.shape)
        print("No D-2 volume for down ",result_index_no_D_2_Volume_Down.shape)
        print("No D-2 prices for up ",result_index_no_D_2_prices_Up.shape)
        print("No D-2 volume for up ",result_index_no_D_2_Volume_Up.shape)
        print("No D-1 and D-2 prices for down ",result_index_no_D_1_D_2_price_Down.shape)
        print("No D-1 and D-2 volume for down ",result_index_no_D_1_D_2_Volume_Down.shape)
        print("No D-1 and D-2 prices for up ",result_index_no_D_1_D_2_prices_Up.shape)
        print("No D-1 and D-2 volume for up ",result_index_no_D_1_D_2_Volume_Up.shape)

    return df_D_1_Down_mod, df_D_2_Down_mod, df_D_1_Up_mod, df_D_2_Up_mod

def master_data_processing(start_date, end_date):
    """
    Function to import and process the input data
    Input: start and end date
    Output: resulting values
    """
    #Import data from API and csv
    df_spot_raw = import_data_API(start_date, end_date, 'spot')
    df_FCRD_up_D1_raw = import_data_API(start_date, end_date, 'FCRD up D-1')
    df_FCRD_up_D2_raw = import_data_API(start_date, end_date, 'FCRD up D-2')
    df_FCRD_up_total_raw = import_data_API(start_date, end_date, 'FCRD up total')
    df_FCRD_down_D1_raw = import_data_API(start_date, end_date, 'FCRD down D-1')
    df_FCRD_down_D2_raw = import_data_API(start_date, end_date, 'FCRD down D-2')
    df_FCRD_down_total_raw = import_data_API(start_date, end_date, 'FCRD down total')
    FCRD_up_act_2022 = pd.read_csv('FCRD_act_data/FCR_D_act_up_2022_row.csv')
    FCRD_up_act_2023 = pd.read_csv('FCRD_act_data/FCR_D_act_up_2023_row.csv')
    FCRD_down_act_2022 = pd.read_csv('FCRD_act_data/FCR_D_act_dn_2022_row.csv')
    FCRD_down_act_2023 = pd.read_csv('FCRD_act_data/FCR_D_act_dn_2023_row.csv')

    #Handing df shape and filling in NaN to missing timesteps
    df_spot = fix_indexing(df_spot_raw, start_date, end_date)
    df_FCRD_up_D1 = fix_indexing(df_FCRD_up_D1_raw, start_date, end_date)
    df_FCRD_up_D2 = fix_indexing(df_FCRD_up_D2_raw, start_date, end_date)
    df_FCRD_up_total = fix_indexing(df_FCRD_up_total_raw, start_date, end_date)
    df_FCRD_down_D1 = fix_indexing(df_FCRD_down_D1_raw, start_date, end_date)
    df_FCRD_down_D2 = fix_indexing(df_FCRD_down_D2_raw, start_date, end_date)
    df_FCRD_down_total = fix_indexing(df_FCRD_down_total_raw, start_date, end_date)

    # correct_index_2022 = pd.date_range(start="2022-01-01", end="2023-01-01", inclusive="left", freq='H')
    # correct_index_2023 = pd.date_range(start="2023-01-01", end="2023-04-01", inclusive="left", freq='H')
    # FCRD_up_act_2022_raw.index = pd.to_datetime(FCRD_up_act_2022_raw['Date'])
    # FCRD_up_act_2022 = FCRD_up_act_2022_raw.reindex(correct_index_2022, fill_value = np.nan)
    # FCRD_up_act_2023_raw.index = pd.to_datetime(FCRD_up_act_2023_raw['Date'])
    # FCRD_up_act_2023 = FCRD_up_act_2023_raw.reindex(correct_index_2023, fill_value = np.nan)
    # FCRD_down_act_2022_raw.index = pd.to_datetime(FCRD_down_act_2022_raw['Date'])
    # FCRD_down_act_2022 = FCRD_down_act_2022_raw.reindex(correct_index_2022, fill_value = np.nan)
    # FCRD_down_act_2023_raw.index = pd.to_datetime(FCRD_down_act_2023_raw['Date'])
    # FCRD_down_act_2023 = FCRD_down_act_2023_raw.reindex(correct_index_2023, fill_value = np.nan)

    #For FCRD, fill in inconsistencies if possible
    df_FCRD_down_D1_v2, df_FCRD_down_D2_v2, df_FCRD_up_D1_v2, df_FCRD_up_D2_v2 = NaN_FCR_D(df_FCRD_down_D1,df_FCRD_down_D2,df_FCRD_down_total,df_FCRD_up_D1,df_FCRD_up_D2,df_FCRD_up_total,printstatement=False)

    #Interpolate and fill NaN values
    df_spot_filled = df_spot.interpolate(method='linear', axis=0)
    df_FCRD_up_D1_filled = df_FCRD_up_D1_v2.interpolate(method='linear', axis=0)
    df_FCRD_up_D2_filled = df_FCRD_up_D2_v2.interpolate(method='linear', axis=0)
    df_FCRD_down_D1_filled = df_FCRD_down_D1_v2.interpolate(method='linear', axis=0)
    df_FCRD_down_D2_filled = df_FCRD_down_D2_v2.interpolate(method='linear', axis=0)
    FCRD_up_act_2022_filled = FCRD_up_act_2022.fillna(0) #Fill missing values with 0
    FCRD_up_act_2023_filled = FCRD_up_act_2023.fillna(0) #Fill missing values with 0
    FCRD_down_act_2022_filled = FCRD_down_act_2022.fillna(0) #Fill missing values with 0
    FCRD_down_act_2023_filled = FCRD_down_act_2023.fillna(0) #Fill missing values with 0

    #Convert to arrays before export
    spot_values = df_spot_filled['SpotPriceEUR'].values
    FCRD_down_D1_values = df_FCRD_down_D1_filled['PriceTotalEUR'].values
    FCRD_down_D2_values = df_FCRD_down_D2_filled['PriceTotalEUR'].values
    FCRD_up_D1_values = df_FCRD_up_D1_filled['PriceTotalEUR'].values
    FCRD_up_D2_values = df_FCRD_up_D2_filled['PriceTotalEUR'].values
    FCRD_up_act_2022_values = FCRD_up_act_2022_filled['Values'].values
    FCRD_up_act_2023_values = FCRD_up_act_2023_filled['Values'].values
    FCRD_down_act_2022_values = FCRD_down_act_2022_filled['Values'].values
    FCRD_down_act_2023_values = FCRD_down_act_2023_filled['Values'].values

    has_missing_up2022 = FCRD_up_act_2022_filled.isna().any()
    has_missing_up2023 = FCRD_up_act_2023_filled.isna().any()
    has_missing_dn2022 = FCRD_down_act_2022_filled.isna().any()
    has_missing_dn2023 = FCRD_down_act_2023_filled.isna().any()
    print(has_missing_up2022)
    print(has_missing_up2023)
    print(has_missing_dn2022)
    print(has_missing_dn2023)


    #Share of D-1 and D-2 by volume
    total_up_volume = df_FCRD_up_D1_filled["PurchasedVolumeTotal"]+df_FCRD_up_D2_filled["PurchasedVolumeTotal"]
    total_down_volume = df_FCRD_down_D1_filled["PurchasedVolumeTotal"]+df_FCRD_down_D2_filled["PurchasedVolumeTotal"]
    FCRD_up_D1_percentage = (df_FCRD_up_D1_filled["PurchasedVolumeTotal"]/total_up_volume).values
    FCRD_up_D2_percentage = (df_FCRD_up_D2_filled["PurchasedVolumeTotal"]/total_up_volume).values
    FCRD_down_D1_percentage = (df_FCRD_down_D1_filled["PurchasedVolumeTotal"]/total_down_volume).values
    FCRD_down_D2_percentage = (df_FCRD_down_D2_filled["PurchasedVolumeTotal"]/total_down_volume).values

    processed_data = {'Spot': spot_values,
                      'FD1_down': FCRD_down_D1_values,
                      'FD2_down': FCRD_down_D2_values,
                      'FD1_up': FCRD_up_D1_values,
                      'FD2_up': FCRD_up_D2_values,
                      'FD_act_up_2022': FCRD_up_act_2022_values,
                      'FD_act_up_2023': FCRD_up_act_2023_values,
                      'FD_act_down_2022': FCRD_down_act_2022_values,
                      'FD_act_down_2023': FCRD_down_act_2023_values,
                      'FD1_up_percentage': FCRD_up_D1_percentage,
                      'FD2_up_percentage': FCRD_up_D2_percentage,
                      'FD1_down_percentage': FCRD_down_D1_percentage,
                      'FD2_down_percentage': FCRD_down_D2_percentage}
    
    return processed_data

def export_to_csv(processed_data):
    """
    Function to export processed data to csv
    Input: model(real, stochastic, learning), processed data
    Output: save csv
    """
    FD_act_up_total = np.concatenate((processed_data['FD_act_up_2022'],processed_data['FD_act_up_2023']))
    FD_act_down_total = np.concatenate((processed_data['FD_act_down_2022'],processed_data['FD_act_down_2023']))

    df = pd.DataFrame(processed_data, columns=['Spot','FD1_down','FD2_down','FD1_up','FD2_up','FD1_up_percentage','FD2_up_percentage','FD1_down_percentage','FD2_down_percentage'])
    df['FD_act_up'] = FD_act_up_total
    df['FD_act_down'] = FD_act_down_total

    df.to_csv("Processed_data/real.csv", sep=',', header=True, index_label='Hour')

# Run following lines to use the functions above:
processed_data = master_data_processing("2022-01-01","2023-04-01")
export_to_csv(processed_data)
print("it worked again!")
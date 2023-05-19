import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import requests

'''
## Run API
If you haven't loaded this before run the API. It will take a little bit to run depending on your computer. (Took me 2 min)
Otherwise if already run before save it in a CSV file and load that file instead.
'''


# This is an API call for FCR-D up and the price areas DK2, SE1 ... SE4

# Check the link for how to "manually" download it.

# https://www.energidataservice.dk/tso-electricity/FcrNdDK2

start_date = "2022-01-01"
end_date = "2023-03-31"
limit = "1000000" # Set limit

ColumnsFCRv2_all = "HourDK, ProductName, PriceArea,AuctionType, PurchasedVolumeLocal, PurchasedVolumeTotal, PriceTotalEUR"
    
# AuctionType = {D-1, D-2 or Total}
# ProductName = {FCR-D upp, FCR-D ned, FCR-N}

dictFilter = {"PriceArea": ("DK2","SE1","SE2","SE3","SE4"),"ProductName": ("FCR-D upp", "FCR-D ned", "FCR-N"),"AuctionType": ("Total", "D-1","D-2")} # To apply filter transform into dict -> json
jsonFilter = json.dumps(dictFilter)
parametersFCRD= {
    "start": start_date,
    "end": end_date,
    "columns": ColumnsFCRv2_all,
    "filter": jsonFilter,
    "limit": limit
    }
response = requests.get(
    url='https://api.energidataservice.dk/dataset/FcrNdDK2', params=parametersFCRD)

result = response.json() # Change to JSON file
records_data = result['records'] # extract the 'records' data
df = pd.DataFrame(records_data) # convert the 'records' data to a pandas DataFrame


print(df)
df.to_csv('FCR_Energinet_Data.csv', index=False) # save the DataFrame as a CSV file

'''
## Load CSV file
If API has already been runned, load the csv instead. 1 second versus 2 minutes ;-) 
'''


df = pd.read_csv('FCR_Energinet_Data.csv')
pd.to_datetime(df["HourDK"]) # Convert HourDK to datetimeprint(df), such that one can use it to index

'''
Check if the summation of all the price areas (except DK1) gives the same as Total
'''

# Is total the same as the summation of DK2, SE1 to SE4?'
lstPriceArea = ["SE1","SE2","SE3","SE4"]
mask = (df["PriceArea"] == "DK2") & (df["AuctionType"] == "Total") & (df["ProductName"] == "FCR-D upp")
sum_pa = df["PurchasedVolumeLocal"].loc[mask].reset_index(drop=True)
for pa in lstPriceArea:
    temp_mask = (df["PriceArea"] == pa) & (df["AuctionType"] == "Total") & (df["ProductName"] == "FCR-D upp")
    sum_pa = sum_pa + df["PurchasedVolumeLocal"].loc[temp_mask].reset_index(drop=True)

print(sum_pa) 
print(df["PurchasedVolumeTotal"].loc[mask].reset_index(drop=True))



'''
## Plot graphs
### All the price area and the volume
'''

product_name = "FCR-D upp"
auction_type = "Total"
price_areas = ["DK2", "SE1", "SE2", "SE3", "SE4"]

plt.figure(figsize=(8, 5))

for area in price_areas:
    mask = (df["PriceArea"] == area) & (df["AuctionType"] == auction_type) & (df["ProductName"] == product_name)
    volume = df["PurchasedVolumeLocal"].loc[mask].reset_index(drop=True)
    plt.plot(volume, label=f"Volume in {area}", drawstyle='steps-post')

mask_total = (df["PriceArea"] == "DK2") & (df["AuctionType"] == auction_type) & (df["ProductName"] == product_name)
xaxis = df["HourDK"].loc[mask_total].reset_index(drop=True)
xaxis = pd.to_datetime(xaxis)# Convert xaxis to datetime
volume_total = df["PurchasedVolumeTotal"].loc[mask_total].reset_index(drop=True)
plt.plot(volume_total, label="Total volume")

# Set the x-axis ticks to be monthly
num_ticks = len(xaxis)
tick_step = max(1, int(np.ceil(num_ticks / 12))) # Show at least one tick per month
month_ticks = xaxis.index % tick_step == 0
if not month_ticks[0]:
    month_ticks[0] = True
if not month_ticks[-1]:
    month_ticks[-1] = True
plt.xticks(xaxis.index[month_ticks], xaxis[month_ticks].dt.strftime("%b %Y"), rotation=45)


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("Purchased Volume in Area [MW]")
plt.show()


'''
## Check the difference between D-1 and D-2 in DK2
'''

product_name = "FCR-D upp"
auction_type = ["D-1", "D-2"]
price_areas = "DK2"
Latest_time = '2023-03-31T00:00:00'
Earliest_time = '2023-03-28T00:00:00'
pd.to_datetime(Latest_time)
pd.to_datetime(Earliest_time)

plt.figure(figsize=(8, 5))


for auction in auction_type:
    mask = (df["HourDK"] >= Earliest_time) & (df["HourDK"] <= Latest_time) & (df["PriceArea"] == price_areas) & (df["AuctionType"] == auction) & (df["ProductName"] == product_name)
    volume = df["PurchasedVolumeLocal"].loc[mask].reset_index(drop=True)
    plt.plot(volume, label=f"Volume in {auction} for DK2", drawstyle='steps-post')


xaxis = df["HourDK"].loc[mask].reset_index(drop=True)
xaxis = pd.to_datetime(xaxis)# Convert xaxis to datetime

# Set the x-axis ticks to be monthly
num_ticks = len(xaxis)
tick_step = max(1, int(np.ceil(num_ticks / 12))) # Show at least one tick per month
month_ticks = xaxis.index % tick_step == 0
if not month_ticks[0]:
    month_ticks[0] = True
if not month_ticks[-1]:
    month_ticks[-1] = True
plt.xticks(xaxis.index[month_ticks], xaxis[month_ticks].dt.strftime("%b %d - %H:%M"), rotation=45)


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("Purchased Volume in Area [MW]")
plt.show()


'''
Check difference between D-1 and D-2 in Total
'''

product_name = "FCR-D upp"
auction_type = ["D-1", "D-2"]
price_areas = "DK2"
Latest_time = '2023-03-31T00:00:00'
Earliest_time = '2023-03-28T00:00:00'
pd.to_datetime(Latest_time)
pd.to_datetime(Earliest_time)

plt.figure(figsize=(8, 5))

for auction in auction_type:
    mask = (df["HourDK"] >= Earliest_time) & (df["HourDK"] <= Latest_time) & (df["PriceArea"] == price_areas) & (df["AuctionType"] == auction) & (df["ProductName"] == product_name)
    volume = df["PurchasedVolumeTotal"].loc[mask].reset_index(drop=True)
    plt.plot(volume, label=f"Total Volume in {auction}", drawstyle='steps-post')

xaxis = df["HourDK"].loc[mask].reset_index(drop=True)
xaxis = pd.to_datetime(xaxis)# Convert xaxis to datetime

# Set the x-axis ticks to be monthly
num_ticks = len(xaxis)
tick_step = max(1, int(np.ceil(num_ticks / 12))) # Show at least one tick per month
month_ticks = xaxis.index % tick_step == 0
if not month_ticks[0]:
    month_ticks[0] = True
if not month_ticks[-1]:
    month_ticks[-1] = True
plt.xticks(xaxis.index[month_ticks], xaxis[month_ticks].dt.strftime("%b %d - %H:%M"), rotation=45)

plt.grid(linestyle='--', color='gray')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("Total Purchased Volume [MW]")
plt.show()


'''
# FCR-D price Analysis
'''
# IT CAN BE SEEN HERE THAT THE PRICE IS THE SAME!!!
product_name = ["FCR-D upp", "FCR-D ned"]
auction_type = "Total"
price_areas = ["DK2", "SE1", "SE2", "SE3", "SE4"]

plt.figure(figsize=(8, 5))

for product in product_name:
    for area in price_areas:
        mask = (df["PriceArea"] == area) & (df["AuctionType"] == auction_type) & (df["ProductName"] == product)
        price = df["PriceTotalEUR"].loc[mask].reset_index(drop=True)
        plt.plot(price, label=f"Price in {area} for {product}", drawstyle='steps-post')

mask_total = (df["PriceArea"] == "DK2") & (df["AuctionType"] == auction_type) & (df["ProductName"] == "FCR-D upp")
xaxis = df["HourDK"].loc[mask_total].reset_index(drop=True)
xaxis = pd.to_datetime(xaxis)# Convert xaxis to datetime

# Set the x-axis ticks to be monthly
num_ticks = len(xaxis)
tick_step = max(1, int(np.ceil(num_ticks / 12))) # Show at least one tick per month
month_ticks = xaxis.index % tick_step == 0
if not month_ticks[0]:
    month_ticks[0] = True
if not month_ticks[-1]:
    month_ticks[-1] = True
plt.xticks(xaxis.index[month_ticks], xaxis[month_ticks].dt.strftime("%b %Y"), rotation=45)


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel("FCR-D Up Reserve price [EUR/MW]")
plt.show()



product_name = ["FCR-D upp", "FCR-D ned"]
auction_type = ["D-1","D-2"]
price_areas = "DK2"

fig, axs = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)

for i, product in enumerate(product_name):
    for j, auction in enumerate(auction_type):
        mask = (df["PriceArea"] == price_areas) & (df["AuctionType"] == auction) & (df["ProductName"] == product)
        price = df["PriceTotalEUR"].loc[mask].reset_index(drop=True)
        axs[j].plot(price, label=f"{product}", drawstyle='steps-post')

        
        axs[j].set_ylabel(f"{auction} Reserve price [EUR/MW]")
        
    mask_total = (df["PriceArea"] == "DK2") & (df["AuctionType"] == "D-1") & (df["ProductName"] == product)
    xaxis = df["HourDK"].loc[mask_total].reset_index(drop=True)
    xaxis = pd.to_datetime(xaxis)# Convert xaxis to datetime

    # Set the x-axis ticks to be monthly
    num_ticks = len(xaxis)
    tick_step = max(1, int(np.ceil(num_ticks / 12))) # Show at least one tick per month
    month_ticks = xaxis.index % tick_step == 0
    if not month_ticks[0]:
        month_ticks[0] = True
    if not month_ticks[-1]:
        month_ticks[-1] = True
    axs[j].set_xticks(xaxis.index[month_ticks])
    axs[j].set_xticklabels(xaxis[month_ticks].dt.strftime("%b %Y"), rotation=45)
axs[0].legend(loc='upper left')
fig.tight_layout()
plt.show()


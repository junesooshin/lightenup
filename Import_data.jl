#General data processing functions

using JSON

function Battery_Specifications(BatteryType)
    # data from the technology catalog
    # https://ens.dk/sites/ens.dk/files/Analyser/technology_data_catalogue_for_energy_storage.pdf
    if BatteryType == "Samsung_SDI_E3_R135_Battery"
        SOC_0 = 3
        E_nom = 6 # [MWh] Energy storage capacity for one unit (2020)
        output_Capacity = 6 # [MW] discharge rate 
        input_Capacity  = 6  # [MW] discharge rate
        Charge_efficiency = 0.98 # [%]
        Discharge_efficiency = 0.97 # [%]
        Specific_investment = 1.042 # [M EUR 2015 / MWh]
        N_cycles = 14000 # number of cycles in a lifetime
        Cost_per_cycle = Specific_investment * E_nom * 10^6 / N_cycles # [EUR/cycle]

        Data = Dict(
            "SOC_0" => SOC_0, "SOC_max" => E_nom, "eta_ch" => Charge_efficiency, 
            "eta_dis" => Discharge_efficiency, "p_dis_max" => output_Capacity,
            "p_ch_max" => input_Capacity, "Cost_per_cycle" => Cost_per_cycle )
        return Data
    end
end

function Define_Training_and_Test_index(d_train_set, d_update = 0)

    #=
    So the data is provided as all of the data for the given year 2022. This function seperate the data such that only the specified period is utilized
    

    d_train. The iteration corresponding to how the index should be constructed. 
        number of days that correspond to the train range
    d_update. Another iteration which go over each test hour, and with that one update the training data to move that with one hour.
        Update. d = 1 -> first day in 2023, d = 2 -> 2nd day, .... d = 89 -> last hour
    index_all_data. All the data foloowing below structure:
    AuctionType. ["D-2", "DA", "D-1"]. Depending on the auctiontype the forecast will change.


    # So the features are data from 2022-01-01 -> 
    # 0, 2022-01-01 00:00:00 , values
    # 1, 2022-01-01 01:00:00 , values
    # ...
    # ...
    # 8760, 2022-12-31 23:00:00 , values
    # ...
    # ...
    # 10920, 2023-03-31 23:00:00 , values 
    
    # This function create indexes which state where a hour and a day is present in the above data set. 
    # The train data is dependent on the month m that has been chosen, and the test indexes are starting from at least the 3rd of january
    
    #    EXAMPLES 
    # d_train = 365 and d_update = 0, AuctionType = D-2
    #     TRAINING_INDEXES                      TEST_INDEXES                        FORECAST_INDEXES
    #     d1  d2  d3 ... ... d365          #     d3                                   #     d1
    # h1  1   25  49         8737          # h1  8809                                 # h1  8806
    # h2  2   26  50         8738          # h2  8810                                 # h2  8807
    # h3  3   27  51         8739          # h3  8811                                 # h3  8808
    # ..                                   # ..                                       # .. 
    # ..                                   # ..                                       # .. 
    # h24 24  48  62         8760          # h24 8832                                 # h24 8829

    # d_train = 5 and d_update = 0, AuctionType = D-2
    #     TRAINING_INDEXES                       TEST_INDEXES                        FORECAST_INDEXES
    #     d1     d2    d3     d4     d5      #     d1                          #       d1
    # h1  8641   8665  8689  8713   8737     # h1  8809                        # h1   8809
    # h2  8642   8666  8690  8714   8738     # h2  8810                        # h2   8810
    # h3  8643   8667  8691  8715   8739     # h3  8811                        # h3   8811
    # ..                                     # ..                              # ..
    # ..                                     # ..                              # .. 
    # h24 8664   8688  8712  8736   8760     # h24 8832                        # h24  8832
    

    # d_train = 5 and d_update = 2, AuctionType = D-2
    #     TRAINING_INDEXES                       TEST_INDEXES                        FORECAST_INDEXES
    #     d1     d2    d3     d4     d5      #     d1                          #       d1
    # h1  8689   8713  8737  8761   8785     # h1  8857                        # h1   8857
    # h2  8690   8714  8738  8762   8786     # h2  8858                        # h2   8858
    # h3  8691   8715  8739  8763   8787     # h3  8859                        # h3   8859
    # ..                                     # ..                              # ..
    # ..                                     # ..                              # .. 
    # h24 8712   8736  8760  8784   8808     # h24 8880                        # h24  8880

    =#
    
    d_train = 365-d_train_set # [m_set = 1, m = 12] And so forth. This is necessary to get a more logical input but the data is set up such that m = 1 means 12 months.
    index_all_data = collect(1:10920)
    Total_Data_length = length(index_all_data) # Total length of the data -> 10920 datapoints

    days_in_2022 = 365        # days, 365
    Hours_in_2022 = days_in_2022 * 24  # hours, 8760
    # println(Hours_in_2022)

    days_in_months_2023 = [31,29,31]
    days_in_2023 = sum(days_in_months_2023[i] for i in 1:3)        # days, 91
    Hours_in_2023 = sum(days_in_months_2023[i] for i in 1:3) * 24  # hours, 2184 
    

    index = collect(1:Total_Data_length) # All the indexes of the whole data set in a list [1,2,3 ... ,10920]
    
    # Here it is the number of data points that fit to:
    # 12 months , 11 months , 10 months , 9 months , 8 months , 7 months , 6 months , 5 months , 4 months , 3 month , 2 months , 1 month 
    
    if (d_train > 365)
        throw(ErrorException("You are trying to retrieve a too long training period!"))
    end
    if (d_update > days_in_2023-2)
        throw(ErrorException("You are trying to retrieve data which is not there!"))
    end  


    ##################################
    ####### CREATE THE INDEXES #######
    ##################################

    #######  TRAINING INDEXES  #######

    start_train_idx = ( d_train + d_update ) * 24 # Retrieve the data point that correspond to the month m and day d
    end_train_idx   = Hours_in_2022 + d_update*24 # Ends at the same training closes to the 2023 data
    Training_period_idx = index[(start_train_idx+1):end_train_idx] # Get the desired index for training

    days_in_training = Int((end_train_idx-start_train_idx)/24)   
    Training_periode_days_hours_idx = reshape(Training_period_idx, (24, days_in_training)) # Reshape it to have an index which can seperate in hours and days
    

    #######    TEST INDEXES    #######

    # WILL ALWAYS START ON THE 3rd of January
    start_test_idx = end_train_idx + 2*24 # start for the test. Which is by default at the 3rd of january(WHICH IS WHY it is +2). May be moved to the right by days for sliding window
    
    # will always give the 24 hours of the testing day.
    end_test_idx = Total_Data_length

    Test_period_idx = index[(start_test_idx+1):end_test_idx]
    days_in_test = Int((end_test_idx-start_test_idx)/24)
    
    Test_periode_days_hours_idx = reshape(Test_period_idx, (24, days_in_test)) # Reshape it to have an index which can seperate in hours and days
    
    #######  FORECAST INDEXES  #######

    start_forecast_idx = d_update*24 + 2*24
    end_forecast_idx = start_forecast_idx + 24
    Forecast_period_idx = index[(start_forecast_idx+1):end_forecast_idx]

    days_in_forecast = Int((end_forecast_idx-start_forecast_idx)/24)
    Forecast_periode_days_hours_idx = reshape(Forecast_period_idx, (24, days_in_forecast)) # Reshape it to have an index which can seperate in hours and days

    ##################################
    #######  OUTPUT FOR MODEL  #######
    ##################################

    D_train = collect(1:days_in_training)          # The total number of days to be iterated over
    D_test  = collect(1:days_in_test)              # The total number of days to be iterated over
    D_forecast = collect(1:days_in_forecast)       # The total number of days to be iterated over
    H       = collect(1:24)                        # The total number of hours to be iterated over

    Data_index = 
        Dict(
        "N_train" => Training_periode_days_hours_idx,
        "N_train_flat" => Training_period_idx,
        "N_test" => Test_periode_days_hours_idx,
        "N_test_flat" => Test_period_idx,
        "N_forecast" => Forecast_periode_days_hours_idx,
        "N_forecast_flat" => Forecast_period_idx, 
        "D_train" => D_train,
        "D_test" => D_test,
        "D_forecast" => D_forecast,
        "H" => H,
        "d_train" => d_train,
        "d_update" => d_update
        )

    return Data_index
end

#idx = Define_Training_and_Test_index(5,0)
#print(idx["D_forecast"])

function Define_Training_and_Test_period(m_set, d = 0, AuctionType = "D-2")

    #=
    So the data is provided as all of the data for the given year 2022. This function seperate the data such that only the specified period is utilized
    
    m_set is transformed to m -> m = 13-m_set. To get the proper logic
    m. The iteration corresponding to how the index should be constructed. 
        m = 1 correspond to 1 month i.e. december in 2022. m = 12 correspond to the whole year
    d. Another iteration which go over each test hour, and with that one update the training data to move that with one hour.
        Update. d = 1 -> first day in 2023, d = 2 -> 2nd day, .... d = 89 -> last hour
    index_all_data. All the data foloowing below structure:
    AuctionType. ["D-2", "DA", "D-1"]. Depending on the auctiontype the forecast will change.


    # So the features are data from 2022-01-01 -> 
    # 0, 2022-01-01 00:00:00 , values
    # 1, 2022-01-01 01:00:00 , values
    # ...
    # ...
    # 8760, 2022-12-31 23:00:00 , values
    # ...
    # ...
    # 10920, 2023-03-31 23:00:00 , values 
    
    # This function create indexes which state where a hour and a day is present in the above data set. 
    # The train data is dependent on the month m that has been chosen, and the test indexes are starting from at least the 3rd of january
    
    #    EXAMPLES 
    # m = 12 and d = 0, AuctionType = D-2
    #     TRAINING_INDEXES                      TEST_INDEXES                        FORECAST_INDEXES
    #     d1  d2  d3 ... ... d365          #     d3    d2    ... ...  d90      #     d1
    # h1  1   25  49         8736          # h1  8809  8833           10897    # h1  8806
    # h2  2   26  50         8737          # h2  8810  8834           10898    # h2  8807
    # h3  3   27  51         8738          # h3  8811  8885           10899    # h3  8808
    # ..                                   # ..                                # .. 
    # ..                                   # ..                                # .. 
    # h24 24  48  62         8760          # h24 8832  8856           10920    # h24 8829

    # or if the file is has m = 1 and d = 0, the indexes correspond to the last month of 2022 and get the indexes which correspond to that in the data
    #     m = 1 and d = 0, AuctionType = D-1                                   
    #     TRAINING_INDEXES                      TEST_INDEXES                       FORECAST_INDEXES
    #     d1     d2    d3   ... ... d31                                        #     d1
    # h1  8016   8041  8065         8736         It is the same as above       # h1  
    # h2  8017   8042  8066         8737                                       # h2
    # h3  8018   8043  8067         8738                                       # h3
    # ..                                                                       # ..
    # ..                                                                       # ..
    # h24 8040   8064  8088         8760                                       # h24

    # or if the file is has m = 1 and d = 3, the indexes correspond to the last month of 2022 but the first three days of the month is removed and the first three days of 2023 (correspond to 72 datapoints) is added and get the indexes which correspond to that in the data
  #     TRAINING_INDEXES                       TEST_INDEXES                        FORECAST_INDEXES
    #     d1     d2    d3   ... ... d31      #     d1                          #     d1
    # h1  8088   8113  8137         8808     # h1  8833                        # h1  
    # h2  8089   8114  8138         8709     # h2  8834                        # h2
    # h3  8090   8115  8139         8710     # h3  8835                        # h3 
    # ..                                     # ..                              # ..
    # ..                                     # ..                              # .. 
    # h24 8112   8136  80           8832     # h24 8856                        # h24
    
    =#
    
    m = 13-m_set # [m_set = 1, m = 12] And so forth. This is necessary to get a more logical input but the data is set up such that m = 1 means 12 months.
    index_all_data = collect(1:10920)
    Total_Data_length = length(index_all_data) # Total length of the data -> 10920 datapoints

    days_in_months_2022 = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    aggregated_days_in_months_2022 = cumsum(days_in_months_2022) # [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    days_in_2022 = sum(days_in_months_2022[i] for i in 1:13)        # days, 365
    Hours_in_2022 = sum(days_in_months_2022[i] for i in 1:13) * 24  # hours, 8760
    # println(Hours_in_2022)

    days_in_months_2023 = [31,29,31]
    aggregated_days_in_months_2023 = cumsum(days_in_months_2023)
    days_in_2023 = sum(days_in_months_2023[i] for i in 1:3)        # days, 91
    Hours_in_2023 = sum(days_in_months_2023[i] for i in 1:3) * 24  # hours, 2184 
    

    index = collect(1:Total_Data_length) # All the indexes of the whole data set in a list [1,2,3 ... ,10920]
    
    # Here it is the number of data points that fit to:
    # 12 months , 11 months , 10 months , 9 months , 8 months , 7 months , 6 months , 5 months , 4 months , 3 month , 2 months , 1 month 
    
    if (m > 12)
        throw(ErrorException("You are trying to retrieve a too long training period!"))
    end
    if (d > days_in_2023-2)
        throw(ErrorException("You are trying to retrieve data which is not there!"))
    end  


    ##################################
    ####### CREATE THE INDEXES #######
    ##################################

    #######  TRAINING INDEXES  #######

    start_train_idx = ( aggregated_days_in_months_2022[m] + d ) * 24 # Retrieve the data point that correspond to the month m and day d
    end_train_idx   = Hours_in_2022 + d*24 # Ends at the same training closes to the 2023 data
    Training_period_idx = index[(start_train_idx+1):end_train_idx] # Get the desired index for training

    days_in_training = Int((end_train_idx-start_train_idx)/24)   
    Training_periode_days_hours_idx = reshape(Training_period_idx, (24, days_in_training)) # Reshape it to have an index which can seperate in hours and days
    

    #######    TEST INDEXES    #######

    # WILL ALWAYS START ON THE 3rd of January
    start_test_idx = end_train_idx + 2*24 # start for the test. Which is by default at the 3rd of january(WHICH IS WHY it is +2). May be moved to the right by days for sliding window
    
    # will always give the 24 hours of the testing day.
    end_test_idx = Total_Data_length

    Test_period_idx = index[(start_test_idx+1):end_test_idx]
    days_in_test = Int((end_test_idx-start_test_idx)/24)
    
    Test_periode_days_hours_idx = reshape(Test_period_idx, (24, days_in_test)) # Reshape it to have an index which can seperate in hours and days
    
    #######  FORECAST INDEXES  #######
    # Lagged with 1,2 and 3 hours!

    # The forecasted indexes correspond to the lagged values.
    if AuctionType == "D-2"
        lagged_hours = 3
    elseif AuctionType == "DA"
        lagged_hours = 2
    elseif AuctionType == "D-1"
        lagged_hours = 1
    end

    start_forecast_idx = start_test_idx - lagged_hours
    end_forecast_idx = Total_Data_length - lagged_hours
    # print(start_forecast_idx, " ", end_forecast_idx)
    Forecast_period_idx = index[(start_forecast_idx+1):end_forecast_idx]
    # print(Forecast_period_idx)

    days_in_forecast = Int((end_forecast_idx-start_forecast_idx)/24)
    Forecast_periode_days_hours_idx = reshape(Forecast_period_idx, (24, days_in_forecast)) # Reshape it to have an index which can seperate in hours and days

    ##################################
    #######  OUTPUT FOR MODEL  #######
    ##################################

    D_train = collect(1:days_in_training)          # The total number of days to be iterated over
    D_test  = collect(1:days_in_test)              # The total number of days to be iterated over
    D_forecast = collect(1:days_in_forecast)       # The total number of days to be iterated over
    H       = collect(1:24)                        # The total number of hours to be iterated over

    Data_index = Dict(
        "N_train" => Training_periode_days_hours_idx,
        "N_train_flat" => Training_period_idx,
        "N_test" => Test_periode_days_hours_idx,
        "N_test_flat" => Test_period_idx,
        "N_forecast" => Forecast_periode_days_hours_idx,
        "N_forecast_flat" => Forecast_period_idx, 
        "D_train" => D_train,
        "D_test" => D_test,
        "D_forecast" => D_forecast,
        "H" => H,
        "m" => m,
        "d" => d)

    return Data_index
end

function load_data(file) 

    dir = dirname(@__FILE__)
    filepath = string(dir , "/Processed_data/real.csv")

    # Load files
    if file == "processed"
        filepath = string(dir , "/Processed_data/" , "real.csv")
        data_import = DataFrame(CSV.File(filepath)) 
        return data_import

    elseif file == "features"
        filepath = string(dir , "/Processed_data/features.csv")
        data_import = DataFrame(CSV.File(filepath)) 
        return data_import

    elseif occursin.("forecast",file) # check if forecast is in file
        filepath = string(dir , "/Processed_data/", file , ".csv")
        data_import_all = DataFrame(CSV.File(filepath))
        data_import = data_import_all[(24*2)+1:end, :] #Cut the first two days to match test data
        return data_import

    elseif file == "forgettingFactor"
        filepath = string(dir , "/Processed_data/", "forgettingFactor.csv")
        data_import = DataFrame(CSV.File(filepath))
        return data_import
    
    end 
end

function construct_acceptance(FCRD_bid, vol_avg_price)
    # Function to calculate the acceptance percentage
    # Input: bid price (as df), volume-weighted average forecast (as vector)
    # Output: acceptance scenarios
    acceptance = zeros(size(FCRD_bid))
    for i in 1:size(FCRD_bid)[1] #hours
        for j in 1:size(FCRD_bid)[2] #days
            if FCRD_bid[i,j] < vol_avg_price[i]
                acceptance[i,j] = 1 #Always accepted if bid is below the volume weighted average
            elseif FCRD_bid[i,j] >= vol_avg_price[i]
                percentage_excess = 1+(FCRD_bid[i,j]-vol_avg_price[i])/vol_avg_price[i]
                acceptance[i,j] = rev_sigmoid(percentage_excess) #Use the reverse sigmoid function
            end
        end
    end
    return acceptance
end

function rev_sigmoid(x)
    #Reverse sigmoid function for PaB acceptance
    # Define the parameters for the sigmoid function
    a = 5
    b = 1
    c = 2
    #reverse sigmoid function
    y = c / (1 + exp(-a*(x-b)))
    y = 2 - y
    return y
end

function save_dict(dict, name)
    
    #dir = dirname(@__FILE__) # The path of the script
    dir = dirname(pwd()) # current directory. Go to where all the thesis folders are present
    filename = "/Results/$(name).json" # The path to the results and the filename
    filepath = string(dir,filename) # The full path
    json_save = JSON.json(dict)
    open(filepath, "w") do f
        write(f, json_save)
    end
end
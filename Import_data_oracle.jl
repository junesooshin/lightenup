#data import function for Oracle model

function data_import_Oracle(data_import, Data_index, test_day)
    #Same input as RT

    #Battery data
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")
    SOC_0 = Data_Battery["SOC_0"]
    SOC_max = Data_Battery["SOC_max"]
    eta_dis = Data_Battery["eta_dis"]
    eta_ch = Data_Battery["eta_ch"]
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    Cost_per_cycle = Data_Battery["Cost_per_cycle"]

    # RT 
    first_h_test = Data_index["N_test"][:,test_day][1]
    last_h_test = Data_index["N_test"][:,test_day][end]
    df_test = data_import[first_h_test:last_h_test, :]

    DA_price_real = df_test[:, "Spot"]
    FD_act_up_real = df_test[:, "FD_act_up"]
    FD_act_down_real = df_test[:, "FD_act_down"]
    FD1_up_price = df_test[:, "FD1_up"]
    FD1_down_price = df_test[:, "FD1_down"]
    FD2_up_price = df_test[:, "FD2_up"]
    FD2_down_price = df_test[:, "FD2_down"]

    
    FD1_down_percentage = ones((24)) # Accepted in accordance to the volume in the market
    FD2_down_percentage = ones((24))
    FD1_up_percentage   = ones((24))
    FD2_up_percentage   = ones((24))
    # ALWAYS ACCEPTED, IF THERE EXIST PROCUMENT IN THE MARKET
    for i in 1:24
        if df_test[i, "FD1_down_percentage"] <= 0 
            FD1_down_percentage[i] = 0     
        end
        
        if df_test[i, "FD2_down_percentage"] <= 0
            FD2_down_percentage[i] = 0      
        end

        if df_test[i, "FD1_up_percentage"] <= 0 
            FD1_up_percentage[i] = 0      
        end

        if df_test[i, "FD2_up_percentage"] <= 0 
            FD2_up_percentage[i] = 0      
        end
    end


    #FD1_down_percentage = repeat([1], 24) # Always accepted
    #FD2_down_percentage = repeat([1], 24)
    #FD1_up_percentage   = repeat([1], 24)
    #FD2_up_percentage   = repeat([1], 24)

    #Export for a selected forecast/test day
    Data = Dict("Time" => [i for i in 1:24], 
                "f_DA_t" => DA_price_real, 
                "f_FD1_up_t" => FD1_up_price, 
                "f_FD2_up_t" => FD2_up_price, 
                "f_FD1_dn_t" => FD1_down_price,
                "f_FD2_dn_t" => FD2_down_price,
                "f_a_up_t" => FD_act_up_real, 
                "f_a_dn_t" => FD_act_down_real, 
                "f_FD1_y_up_t" => FD1_down_percentage,
                "f_FD1_y_dn_t" => FD2_down_percentage,
                "f_FD2_y_up_t" => FD1_up_percentage,
                "f_FD2_y_dn_t" => FD2_up_percentage,
                #Battery constraints 
                "SOC_0" => SOC_0, 
                "SOC_max" => SOC_max, 
                "eta_dis" => eta_dis,
                "eta_ch" => eta_ch,
                "p_dis_max" => p_dis_max,
                "p_ch_max" => p_ch_max, 
                "Cost_per_cycle" => Cost_per_cycle)
    
    @info("Data import for oracle model complete!")
    return Data
end
#data import functions for real-time operation model

function data_import_real(data_import, Data_index, test_day, Bid_Results)
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")

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
    FD1_down_percentage = df_test[:, "FD1_down_percentage"]
    FD2_down_percentage = df_test[:,"FD2_down_percentage"]
    FD1_up_percentage   = df_test[:, "FD1_up_percentage"]
    FD2_up_percentage   = df_test[:, "FD2_up_percentage"]

    FD1_up_bid_price = Bid_Results["f_lambda_FD1_up"]
    FD1_down_bid_price = Bid_Results["f_lambda_FD1_dn"]
    FD2_up_bid_price = Bid_Results["f_lambda_FD2_up"]
    FD2_down_bid_price = Bid_Results["f_lambda_FD2_dn"]

    #Construct acceptance
    # The acceptance is a function of whether the bid is equal or below the true price. OR whether there are even bought volumes in the market. 
    acceptance_FD1_up   = zeros((24))
    acceptance_FD1_down = zeros((24))
    acceptance_FD2_up   = zeros((24))
    acceptance_FD2_down = zeros((24))
    
    acceptance_criteria_factor = 1.05

    for i in 1:24
        if FD1_up_percentage[i] > 0 && FD1_up_bid_price[i] <= FD1_up_price[i]*acceptance_criteria_factor
            acceptance_FD1_up[i] = 1      
        end
        
        if FD1_down_percentage[i] > 0 && FD1_down_bid_price[i] <= FD1_down_price[i]*acceptance_criteria_factor
            acceptance_FD1_down[i] = 1      
        end

        if FD2_down_percentage[i] > 0 && FD2_down_bid_price[i] <= FD2_down_price[i]*acceptance_criteria_factor
            acceptance_FD2_down[i] = 1      
        end

        if FD2_up_percentage[i] > 0 && FD2_up_bid_price[i] <= FD2_up_price[i]*acceptance_criteria_factor
            acceptance_FD2_up[i] = 1      
        end
    end

    #acceptance_FD1_up = [FD1_up_bid_price[i] <= FD1_up_price[i] ? 1 : 0 for i in 1:24]
    #acceptance_FD1_down = [FD1_down_bid_price[i] <= FD1_down_price[i] ? 1 : 0 for i in 1:24]
    #acceptance_FD2_up = [FD2_up_bid_price[i] <= FD2_up_price[i] ? 1 : 0 for i in 1:24]
    #acceptance_FD2_down = [FD2_down_bid_price[i] <= FD2_down_price[i] ? 1 : 0 for i in 1:24]

    DA_up_bid = Bid_Results["b_DA_up"]
    DA_down_bid = Bid_Results["b_DA_dn"]
    FD1_up_bid = Bid_Results["b_FD1_up"].*acceptance_FD1_up
    FD1_down_bid = Bid_Results["b_FD1_dn"].*acceptance_FD1_down
    FD2_up_bid = Bid_Results["b_FD2_up"].*acceptance_FD2_up
    FD2_down_bid = Bid_Results["b_FD2_dn"].*acceptance_FD2_down

    Data = Dict("T" => [i for i in 1:24],
                # Realizations
                "DA_price_t" => DA_price_real,
                "FD_a_up_t" => FD_act_up_real,
                "FD_a_dn_t" => FD_act_down_real,
                "FD1_up_price" => FD1_up_price,
                "FD1_down_price" => FD1_down_price,
                "FD2_up_price" => FD2_up_price,
                "FD2_down_price" => FD2_down_price,

                # acceptance 
                "acceptance_FD1_up" => acceptance_FD1_up,
                "acceptance_FD1_down" => acceptance_FD1_down, 
                "acceptance_FD2_up" => acceptance_FD2_up,
                "acceptance_FD2_down" => acceptance_FD2_down,

                # Bids
                "DA_up_bid" => DA_up_bid,
                "DA_down_bid" => DA_down_bid,
                "FD1_up_bid" => FD1_up_bid,
                "FD1_down_bid" => FD1_down_bid,
                "FD2_up_bid" => FD2_up_bid,
                "FD2_down_bid" => FD2_down_bid,
                "FD1_up_bid_price" => FD1_up_bid_price,
                "FD1_down_bid_price" => FD1_down_bid_price,
                "FD2_up_bid_price" => FD2_up_bid_price,
                "FD2_down_bid_price" => FD2_down_bid_price,

                # Technical
                "SOC_0" => Data_Battery["SOC_0"], 
                "SOC_max" => Data_Battery["SOC_max"], 
                "eta_dis" => Data_Battery["eta_dis"],
                "eta_ch" => Data_Battery["eta_ch"],
                "p_dis_max" => Data_Battery["p_dis_max"],
                "p_ch_max" => Data_Battery["p_ch_max"], 
                "Cost_per_cycle" => Data_Battery["Cost_per_cycle"],
                "penalty" => 1000) 
    return Data
end
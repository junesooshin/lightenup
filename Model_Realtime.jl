
function RT_operation(Data)
    #Parameters 
    T = Data["T"]
    SOC_0 = Data["SOC_0"] 
    SOC_max = Data["SOC_max"] 
    eta_dis = Data["eta_dis"] 
    eta_ch = Data["eta_ch"] 
    p_dis_max = Data["p_dis_max"] 
    p_ch_max = Data["p_ch_max"]
    Cost_per_cycle = Data["Cost_per_cycle"]
    penalty = Data["penalty"]

    #Real values on the test day
    DA_price = Data["DA_price_t"]
    FD_a_up_t = Data["FD_a_up_t"]
    FD_a_dn_t = Data["FD_a_dn_t"]

    #Accepted bids
    DA_up_bid = Data["DA_up_bid"]
    DA_dn_bid = Data["DA_down_bid"]
    FD1_up_bid = Data["FD1_up_bid"]
    FD1_dn_bid = Data["FD1_down_bid"] 
    FD2_up_bid = Data["FD2_up_bid"] 
    FD2_dn_bid = Data["FD2_down_bid"] 

    #Corresponding bid prices for FCRD
    FD1_up_bid_price = Data["FD1_up_bid_price"][:,1]
    FD1_dn_bid_price = Data["FD1_down_bid_price"][:,1]
    FD2_up_bid_price = Data["FD2_up_bid_price"][:,1]
    FD2_dn_bid_price = Data["FD2_down_bid_price"][:,1]

    SOC = zeros(24)
    p_all_up = zeros(24)
    p_all_dn = zeros(24)
    failure_up = zeros(24)
    failure_dn = zeros(24)
    total_penalty = zeros(24)

    R_FD2 = sum(FD2_up_bid_price[t]*FD2_up_bid[t] + FD2_dn_bid_price[t]*FD2_dn_bid[t] for t in T)
    R_FD2_t = FD2_up_bid_price.*FD2_up_bid + FD2_dn_bid_price.*FD2_dn_bid
    R_DA = sum(DA_price[t]*(DA_up_bid[t] - DA_dn_bid[t]) for t in T)
    R_DA_t = DA_price.*(DA_up_bid - DA_dn_bid)
    R_FD1 = sum(FD1_up_bid_price[t]*FD1_up_bid[t] + FD1_dn_bid_price[t]*FD1_dn_bid[t] for t in T)
    R_FD1_t = FD1_up_bid_price.*FD1_up_bid + FD1_dn_bid_price.*FD1_dn_bid

    for t in T
        p_all_up[t] = DA_up_bid[t] + FD_a_up_t[t]*(FD2_up_bid[t]+FD1_up_bid[t])  
        p_all_dn[t] = DA_dn_bid[t] + FD_a_dn_t[t]*(FD2_dn_bid[t]+FD1_dn_bid[t])
        p_dis_max < p_all_up[t] ? failure_up[t] = p_all_up[t] - p_dis_max : failure_up[t] = 0
        p_ch_max < p_all_dn[t] ? failure_dn[t] = p_all_dn[t] - p_ch_max : failure_dn[t] = 0
        if t == 1
            SOC[t] = SOC_0 + eta_ch*p_all_dn[t] - eta_dis*p_all_up[t]
        else
            SOC[t] = SOC[t-1] + eta_ch*p_all_dn[t] - eta_dis*p_all_up[t]
        end
    end

    C_Deg = sum(p_all_up[t] + p_all_dn[t] for t in T)/(2*SOC_max) * Cost_per_cycle
    C_Deg_t = (p_all_up + p_all_dn)./(2*SOC_max) .* Cost_per_cycle
    penalty = penalty*sum(failure_up[t] + failure_dn[t] for t in T)
    penalty_t = penalty.*(failure_up + failure_dn) 
    profit = R_FD2 + R_DA + R_FD1 - C_Deg - penalty
    profit_t = R_FD2_t + R_DA_t + R_FD1_t - C_Deg_t - penalty_t

    RT_results = Dict(#Results from RT operation
                      "p_dis_max" => p_dis_max,
                      "p_ch_max" => p_ch_max,
                      "p_all_up" => p_all_up,
                      "p_all_dn" => p_all_dn,
                      "SOC" => SOC,
                      "failure_up" => failure_up,
                      "failure_dn" => failure_dn,
                      "R_FD2" => R_FD2,
                      "R_FD2_t" => R_FD2_t,
                      "R_DA" => R_DA,
                      "R_DA_t" => R_DA_t,
                      "R_FD1" => R_FD1,
                      "R_FD1_t" => R_FD1_t,
                      "C_Deg" => C_Deg,
                      "C_Deg_t" => C_Deg_t,
                      "penalty" => penalty,
                      "penalty_t" => penalty_t,
                      "profit" => profit,
                      "profit_t" => profit_t,
                      #Real activation
                      "FD_a_up_t" => FD_a_up_t,
                      "FD_a_dn_t" => FD_a_dn_t,
                      # Real prices
                      "DA_price" => DA_price,
                      "FD1_up_price_test" => Data["FD1_up_price"],
                      "FD1_down_price_test" => Data["FD1_down_price"],
                      "FD2_up_price_test" => Data["FD2_up_price"],
                      "FD2_down_price_test" => Data["FD2_down_price"],
                      # bid prices
                      "FD1_up_bid_price" => FD1_up_bid_price,
                      "FD1_dn_bid_price" => FD1_dn_bid_price,
                      "FD2_up_bid_price" => FD2_up_bid_price,
                      "FD2_dn_bid_price" => FD2_dn_bid_price,
                      # acceptance 
                      "acceptance_FD1_up" => Data["acceptance_FD1_up"],
                      "acceptance_FD1_down" => Data["acceptance_FD1_down"], 
                      "acceptance_FD2_up" => Data["acceptance_FD2_up"],
                      "acceptance_FD2_down" => Data["acceptance_FD2_down"]
                      )

  return RT_results
end

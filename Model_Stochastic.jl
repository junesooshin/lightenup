
function stochastic_model(Data)
    #Parameters independent of stages
    T = Data["T"]
    T2 = T[2:end]
    t_ini = T[1]
    W1 = [i for i in 1:Data["size_W1"]]
    W2 = [i for i in 1:Data["size_W2"]]
    W3 = [i for i in 1:Data["size_W3"]]
    pi1 = repeat([1/length(W1)], length(W1))
    pi2 = repeat([1/length(W2)], length(W2))
    pi3 = repeat([1/length(W3)], length(W3))
    SOC_0 = Data["SOC_0"] 
    SOC_max = Data["SOC_max"] 
    eta_dis = Data["eta_dis"] 
    eta_ch = Data["eta_ch"] 
    p_dis_max = Data["p_dis_max"] 
    p_ch_max = Data["p_ch_max"]
    Cost_per_cycle = Data["Cost_per_cycle"]

    #2nd stage parameters [t,w1]
    f_FD2_up_tw = Data["f_FD2_up_tw"] 
    f_FD2_dn_tw = Data["f_FD2_dn_tw"] 
    f_FD2_y_up_tw = Data["f_FD2_y_up_tw"] 
    f_FD2_y_dn_tw = Data["f_FD2_y_dn_tw"] 

    #3rd stage parameters [t,w1,w2]
    f_DA_tw     = Data["f_DA_tw"] 

    #4th stage parameters [t,w1,w2,w3]
    f_FD1_up_tw = Data["f_FD1_up_tw"]  
    f_FD1_dn_tw = Data["f_FD1_dn_tw"] 
    f_FD1_y_up_tw = Data["f_FD1_y_up_tw"] 
    f_FD1_y_dn_tw = Data["f_FD1_y_dn_tw"] 
    f_a_up_tw   = Data["f_a_up_tw"] 
    f_a_dn_tw   = Data["f_a_dn_tw"] 

    m_sto = Model(Gurobi.Optimizer)

    #1st stage decision variable [t]
    @variable(m_sto, b_FD2_up[T] >= 0)  
    @variable(m_sto, b_FD2_dn[T] >= 0)   

    #2nd stage decision variable [t,w1]
    @variable(m_sto, p_FD2_up[T,W1] >= 0)            
    @variable(m_sto, p_FD2_dn[T,W1] >= 0)   
    @variable(m_sto, p_DA_up[T,W1] >= 0)        
    @variable(m_sto, p_DA_dn[T,W1] >= 0)  

    #3rd stage decision variable [t,w1,w2]  
    @variable(m_sto, b_FD1_up[T,W1,W2] >= 0)              
    @variable(m_sto, b_FD1_dn[T,W1,W2] >= 0)   

    #4th stage decision variable [t,w1,w2,w3]
    @variable(m_sto, p_FD1_up[T,W1,W2,W3] >= 0)           
    @variable(m_sto, p_FD1_dn[T,W1,W2,W3] >= 0)                
    @variable(m_sto, p_all_dn[T,W1,W2,W3] >= 0)             
    @variable(m_sto, p_all_up[T,W1,W2,W3] >= 0)           
    @variable(m_sto, SOC[T,W1,W2,W3] >= 0)                 

    #= # THIS ONE IS WEIRD!! IF YOU IMPLEMENT THESE THE VALUE WILL DECREASE DRAMATICALLY????
    #Stage independent variable (revenues and costs at each stage)
    @variable(m_sto, G_FD2)
    @variable(m_sto, G_DA)                      
    @variable(m_sto, G_FD1)             
    @variable(m_sto, C_Deg)                       

    @objective(m_sto, Max, G_FD2 + G_DA + G_FD1 - C_Deg )

    @constraint(m_sto, G_FD2 == sum(pi1[w1]*(f_FD2_up_tw[t,w1]*p_FD2_up[t,w1]
                                            + f_FD2_dn_tw[t,w1]*p_FD2_dn[t,w1]) 
                                            for t in T, w1 in W1))
    @constraint(m_sto, G_DA == sum(pi1[w1]*pi2[w2]*(f_DA_tw[t,w1,w2]
                                                    *(p_DA_up[t,w1]-p_DA_dn[t,w1])) 
                                                    for t in T, w1 in W1, w2 in W2)) 
    @constraint(m_sto, G_FD1 == sum(pi1[w1]*pi2[w2]*pi3[w3]*(f_FD1_up_tw[t,w1,w2,w3]*p_FD1_up[t,w1,w2,w3]  
                                                            + f_FD1_dn_tw[t,w1,w2,w3]*p_FD1_dn[t,w1,w2,w3]) 
                                                            for t in T, w1 in W1, w2 in W2, w3 in W3)) 
    @constraint(m_sto, C_Deg == sum(pi1[w1]*pi2[w2]*pi3[w3]*(p_all_dn[t,w1,w2,w3] + p_all_up[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3)/(2*SOC_max) * Cost_per_cycle) # Constraint to set G_Bal
    =#  

    #Stage independent variable (revenues and costs at each stage)
    @variable(m_sto, G_FD2[t in T] >= 0)
    @variable(m_sto, G_DA[t in T] >= 0)                      
    @variable(m_sto, G_FD1[t in T] >= 0)             
    @variable(m_sto, C_Deg[t in T] >= 0)                        

    @objective(m_sto, Max, sum(G_FD2[t] + G_DA[t] + G_FD1[t] - C_Deg[t] for t in T) )

    @constraint(m_sto, Gain_FD2[t in T], G_FD2[t] == sum(pi1[w1]*(f_FD2_up_tw[t,w1]*p_FD2_up[t,w1]
                                            + f_FD2_dn_tw[t,w1]*p_FD2_dn[t,w1]) 
                                            for w1 in W1))
    @constraint(m_sto, Gain_DA[t in T], G_DA[t] == sum(pi1[w1]*pi2[w2]*(f_DA_tw[t,w1,w2]
                                                    *(p_DA_up[t,w1]-p_DA_dn[t,w1])) 
                                                    for w1 in W1, w2 in W2)) 
    @constraint(m_sto, Gain_FD1[t in T], G_FD1[t] == sum(pi1[w1]*pi2[w2]*pi3[w3]*(f_FD1_up_tw[t,w1,w2,w3]*p_FD1_up[t,w1,w2,w3]  
                                                            + f_FD1_dn_tw[t,w1,w2,w3]*p_FD1_dn[t,w1,w2,w3]) 
                                                            for w1 in W1, w2 in W2, w3 in W3)) 
    @constraint(m_sto, Deg[t in T], C_Deg[t] == sum(pi1[w1]*pi2[w2]*pi3[w3]*(p_all_dn[t,w1,w2,w3] + p_all_up[t,w1,w2,w3]) for w1 in W1, w2 in W2, w3 in W3)/(2*SOC_max) * Cost_per_cycle) # Constraint to set G_Bal



    @constraint(m_sto, p_all_dn_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_all_dn[t,w1,w2,w3] == p_DA_dn[t,w1] + f_a_dn_tw[t,w1,w2,w3]*(p_FD1_dn[t,w1,w2,w3] + p_FD2_dn[t,w1])) # Constraint to keep track of the summation of all the charge
    @constraint(m_sto, p_all_up_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_all_up[t,w1,w2,w3] == p_DA_up[t,w1] + f_a_up_tw[t,w1,w2,w3]*(p_FD1_up[t,w1,w2,w3] + p_FD2_up[t,w1])) # Constraint to keep track of the summation of all the charge

    @constraint(m_sto, SOC_con[t in T2, w1 in W1, w2 in W2, w3 in W3], SOC[t,w1,w2,w3] == SOC[t-1,w1,w2,w3] + eta_ch*p_all_dn[t,w1,w2,w3] - eta_dis*p_all_up[t,w1,w2,w3] ) # Constraint State of charge

    @constraint(m_sto, SOC_con_ini[w1 in W1, w2 in W2, w3 in W3], SOC[t_ini,w1,w2,w3] == SOC_0 + eta_ch*p_all_dn[t_ini,w1,w2,w3] - eta_dis*p_all_up[t_ini,w1,w2,w3] ) # Constraint State of charge

    @constraint(m_sto, SOC_cap_con[t in T, w1 in W1, w2 in W2, w3 in W3], SOC[t,w1,w2,w3] >= (p_DA_up[t,w1] + b_FD1_up[t,w1,w2] + b_FD2_up[t]) ) # To ensure that enough energy in battery for upregulation/discharging. The SOC need to be bigger or equal to all the bids combined for that hour
    @constraint(m_sto, SOC_cap_con2[t in T, w1 in W1, w2 in W2, w3 in W3], SOC[t,w1,w2,w3] <= SOC_max - (p_DA_dn[t,w1] + b_FD1_dn[t,w1,w2] + b_FD2_dn[t]) ) # To ensure that enough energy can be downregulated/charged to the battery. The SOC need to be smaller or equal to the max SOC minus all the downregulated bids combined for that hour

    @constraint(m_sto, Charging_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_DA_dn[t,w1] + b_FD1_dn[t,w1,w2] + b_FD2_dn[t] <= p_ch_max ) # Constraint State of charge
    @constraint(m_sto, Discharging_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_DA_up[t,w1] + b_FD1_up[t,w1,w2] + b_FD2_up[t] <= p_dis_max ) # Constraint State of charge

    @constraint(m_sto, FD1_up_acc_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_FD1_up[t,w1,w2,w3] == f_FD1_y_up_tw[t,w1,w2,w3]*b_FD1_up[t,w1,w2] ) # The true power after corrected for acceptance of the bid
    @constraint(m_sto, FD2_up_acc_con[t in T, w1 in W1], p_FD2_up[t,w1] == f_FD2_y_up_tw[t,w1]*b_FD2_up[t] ) # The true power after corrected for acceptance of the bid
    @constraint(m_sto, FD1_dn_acc_con[t in T, w1 in W1, w2 in W2, w3 in W3], p_FD1_dn[t,w1,w2,w3] == f_FD1_y_dn_tw[t,w1,w2,w3]*b_FD1_dn[t,w1,w2] ) # The true power after corrected for acceptance of the bid
    @constraint(m_sto, FD2_dn_acc_con[t in T, w1 in W1], p_FD2_dn[t,w1] == f_FD2_y_dn_tw[t,w1]*b_FD2_dn[t] ) # The true power after corrected for acceptance of the bid

    optimize!(m_sto)

    @info("Stochastic model terminates with status: $(termination_status(m_sto))")
    if termination_status(m_sto) == MOI.OPTIMAL
        sto_solution = Dict(
            # Input
            "f_FD2_up_tw_input" => f_FD2_up_tw[T,W1],
            "f_FD2_dn_tw_input" => f_FD2_dn_tw[T,W1],
            "f_DA_tw_input" => f_DA_tw[T,W1,W2],
            "f_FD1_up_tw_input" => f_FD1_up_tw[T,W1,W2,W3],
            "f_FD1_dn_tw_input" => f_FD1_dn_tw[T,W1,W2,W3],
            "f_a_up_tw_input"=> f_a_up_tw[T,W1,W2,W3],
            "f_a_dn_tw_input"=> f_a_dn_tw[T,W1,W2,W3],
            "f_FD1_y_up_tw_input" => f_FD1_y_up_tw[T,W1,W2,W3],
            "f_FD1_y_dn_tw_input" => f_FD1_y_dn_tw[T,W1,W2,W3],
            "f_FD2_y_up_tw_input" => f_FD2_y_up_tw[T,W1],
            "f_FD2_y_dn_tw_input" => f_FD2_y_dn_tw[T,W1],

            # Output 
            # "comp_time" => MOI.get(m_sto, MOI.SolveTime()), # Does not work for Daniel
            "b_FD2_up" => [value.(b_FD2_up[t]) for t in T],
            "b_FD2_dn" => [value.(b_FD2_dn[t]) for t in T],
            "p_FD2_up" => [value.(p_FD2_up[t,w1]) for t in T, w1 in W1],
            "p_FD2_dn" => [value.(p_FD2_dn[t,w1]) for t in T, w1 in W1],
            "p_DA_up" => [value.(p_DA_up[t,w1]) for t in T, w1 in W1],
            "p_DA_dn" => [value.(p_DA_dn[t,w1]) for t in T, w1 in W1],
            "b_FD1_up" => [value.(b_FD1_up[t,w1,w2]) for t in T, w1 in W1, w2 in W2],
            "b_FD1_dn" => [value.(b_FD1_dn[t,w1,w2]) for t in T, w1 in W1, w2 in W2],
            "p_FD1_up" => [value.(p_FD1_up[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3],
            "p_FD1_dn" => [value.(p_FD1_dn[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3],
            "p_all_dn" => [value.(p_all_dn[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3],
            "p_all_up" => [value.(p_all_up[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3],
            "SOC" => [value.(SOC[t,w1,w2,w3]) for t in T, w1 in W1, w2 in W2, w3 in W3],
            "G_FD2" => [value.(G_FD2[t]) for t in T],
            "G_DA" => [value.(G_DA[t]) for t in T],
            "G_FD1" => [value.(G_FD1[t]) for t in T],
            "C_Deg" => [value.(C_Deg[t]) for t in T],
            #Input
            "SOC_max" => SOC_max,
            "Cost_per_cycle" => Cost_per_cycle)
        @info("Solved Stochastic model!")
        return sto_solution
    else
        @info("No optimal solution available")
    end
end

function create_bid_stochastic(Data, sto_solution)
    size_W1 = size(Data["f_FD1_up_tw"])[2]
    size_W2 = size(Data["f_FD1_up_tw"])[3]
    size_W3 = size(Data["f_FD1_up_tw"])[4]

    b_FD2_up = sto_solution["b_FD2_up"]
    b_FD2_dn = sto_solution["b_FD2_dn"]
    p_FD2_up = mean(sto_solution["p_FD2_up"], dims=2)[:,1]
    p_FD2_dn = mean(sto_solution["p_FD2_dn"], dims=2)[:,1]
    b_DA_up = mean(sto_solution["p_DA_up"], dims=2)[:,1]
    b_DA_dn = mean(sto_solution["p_DA_dn"], dims=2)[:,1]
    b_FD1_up = mean(reshape(sto_solution["b_FD1_up"], (24, size_W1*size_W2)), dims=2)[:,1]
    b_FD1_dn = mean(reshape(sto_solution["b_FD1_dn"], (24, size_W1*size_W2)), dims=2)[:,1]
    p_FD1_up = mean(reshape(sto_solution["p_FD1_up"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    p_FD1_dn = mean(reshape(sto_solution["p_FD1_dn"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    p_all_up = mean(reshape(sto_solution["p_all_up"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    p_all_dn = mean(reshape(sto_solution["p_all_dn"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]

    soc = mean(reshape(sto_solution["SOC"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    f_FD1_y_up_t = mean(reshape(sto_solution["f_FD1_y_up_tw_input"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    f_FD1_y_dn_t = mean(reshape(sto_solution["f_FD1_y_dn_tw_input"], (24, size_W1*size_W2*size_W3)), dims=2)[:,1]
    f_FD2_y_up_t = mean(sto_solution["f_FD2_y_up_tw_input"], dims=2)[:,1]
    f_FD2_y_dn_t = mean(sto_solution["f_FD2_y_dn_tw_input"], dims=2)[:,1]

    #f_lambda_FD2_up = mean(Data["f_FD2_up_tw"], dims=2)[:,1]
    #f_lambda_FD2_dn = mean(Data["f_FD2_dn_tw"], dims=2)[:,1]
    #f_lambda_FD1_up = mean(reshape(Data["f_FD1_up_tw"], (24,size_W1*size_W2*size_W3)), dims=2)[:,1]
    #f_lambda_FD1_dn = mean(reshape(Data["f_FD1_dn_tw"], (24,size_W1*size_W2*size_W3)), dims=2)[:,1]

    ################ The Forecasted prices i.e. the bid prices ################

    f_DA = Data["f_Spot_price"] #Already selected for forecast day
    f_lambda_FD2_up = Data["f_FD2_up"][:,1] #Flatten array
    f_lambda_FD2_dn = Data["f_FD2_dn"][:,1] #Flatten array
    f_lambda_FD1_up = Data["f_FD1_up"][:,1] #Flatten array
    f_lambda_FD1_dn = Data["f_FD1_dn"][:,1] #Flatten array

    #Calculate expected revenue
    G_FD2_t = f_lambda_FD2_up.*p_FD2_up .+ f_lambda_FD2_dn.*p_FD2_dn 
    G_FD1_t = f_lambda_FD1_up.*p_FD1_up .+ f_lambda_FD1_dn.*p_FD1_dn 
    G_DA_t = f_DA.*(b_DA_up .- b_DA_dn) 
    C_Deg_t = (p_all_dn .+ p_all_up)./(2*sto_solution["SOC_max"]) .* sto_solution["Cost_per_cycle"] 
    obj_t = G_FD2_t + G_FD1_t + G_DA_t - C_Deg_t
    
    Bid_Results = Dict(
                    "obj_t" => obj_t[:,1], #Flatten array
                    "G_FD2" => sto_solution["G_FD2"],
                    "G_DA" => sto_solution["G_DA"],
                    "G_FD1"=> sto_solution["G_FD1"],
                    "C_Deg" => sto_solution["C_Deg"],
                    "b_FD2_up" => b_FD2_up, 
                    "b_FD2_dn" => b_FD2_dn,
                    "b_DA_up"  => b_DA_up , 
                    "b_DA_dn"  => b_DA_dn,
                    "b_FD1_up" => b_FD1_up, 
                    "b_FD1_dn" => b_FD1_dn,
                    "f_lambda_FD2_up" => f_lambda_FD2_up, 
                    "f_lambda_FD2_dn" => f_lambda_FD2_dn,
                    "f_lambda_FD1_up" => f_lambda_FD1_up, 
                    "f_lambda_FD1_dn" => f_lambda_FD1_dn,
                    "f_DA_t" => f_DA,
                    "SOC" => soc,
                    "f_FD1_y_up_t" => f_FD1_y_up_t,
                    "f_FD1_y_dn_t" => f_FD1_y_dn_t,
                    "f_FD2_y_up_t" => f_FD2_y_up_t,
                    "f_FD2_y_dn_t" => f_FD2_y_dn_t,
                    #Inputs
                    "f_FD2_up_tw_input" => sto_solution["f_FD2_up_tw_input"],
                    "f_FD2_dn_tw_input" => sto_solution["f_FD2_dn_tw_input"],
                    "f_DA_tw_input" => sto_solution["f_DA_tw_input"],
                    "f_FD1_up_tw_input" => sto_solution["f_FD1_up_tw_input"],
                    "f_FD1_dn_tw_input" => sto_solution["f_FD1_dn_tw_input"],
                    "f_a_up_tw_input" => sto_solution["f_a_up_tw_input"],
                    "f_a_dn_tw_input" => sto_solution["f_a_dn_tw_input"],
                    "f_FD1_y_up_tw_input" => sto_solution["f_FD1_y_up_tw_input"],
                    "f_FD1_y_dn_tw_input" => sto_solution["f_FD1_y_dn_tw_input"],
                    "f_FD2_y_up_tw_input" => sto_solution["f_FD2_y_up_tw_input"],
                    "f_FD2_y_dn_tw_input" => sto_solution["f_FD2_y_dn_tw_input"]
                    )
    
    @info("New stochastic solution saved!")
    return Bid_Results
end
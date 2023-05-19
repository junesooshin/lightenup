import seaborn as sns
import matplotlib.pyplot as plt


def plot_data_structure():


    Method =                        ["1 m / 8 d upd","1 m / 1 d upd","2 m / 8 d upd","2 m / 1 d upd", "                ","12 m / 8 d upd","12 m / 4 d upd","12 m / 2 d upd","12 m / 1 d upd","   ","1 m / no upd", "2 m / no upd", "                ", "11 m / no upd", "12 m / no upd", ""]
    Start =                         [300-30+3*8,      300-30+3*1,     300-30*2+3*8,      300-30*2+3*1,           0,                      3*8,         3*4,            3*2,         3,          0,      300-30,         300-60,                0,                30,                0,          0]
    Length_of_training =            [30,              30,                   30*2,              30*2,            0,                      300,         300,          300,         300,          0,         30,           60,                 0,               300-30,           300,          0]
    Lag_between_train_test =        [2,               2,                     2,               2,              0,                       2,           2,           2,           2,           0,          2,            2,                 0,                 2,                2,          0] # Starts 3rd of january
    Length_of_test =                [3,               3,                     3,               3,            0,                       3,           3,           3,           3,           0,          3,            3,                 0,                 3,                3,          0]


    fig, ax = plt.subplots(figsize=(7.5, 5))

    sns.set_style('whitegrid')

    b0 = plt.barh(Method, Start, color="#c2c2c2")
    b1 = plt.barh(Method, Length_of_training, left=Start, color="#1f77b4")
    b2 = plt.barh(Method, Lag_between_train_test, left=[Start[i]+Length_of_training[i] for i in range(len(Method))], color="#c2c2c2")
    b3 = plt.barh(Method, Length_of_test, left=[Start[i]+Length_of_training[i]+Lag_between_train_test[i] for i in range(len(Method))], color="#ff7f0e")

    ax.set_title('Training Schedule for two days prior to operation/test')

    ax.tick_params(axis='y', labelsize=8)

    legend_labels = ['Training', 'Testing', 'Not used']
    ax.legend((b1, b3, b2), legend_labels,loc='lower center',bbox_to_anchor=(0.5,-0.05), ncol= 3)

    plt.yticks(rotation=0)
    plt.tick_params(axis='y', which='both', length=0)
    sns.despine()
    ax.spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.show()
    
plot_data_structure()
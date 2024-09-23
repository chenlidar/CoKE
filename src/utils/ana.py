from utils.method import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_proportion(df1, df2, ax, label_list, color_list, marker_list):
    for df, label, color, marker in zip([df1, df2], label_list, color_list, marker_list):
        df['a_bins'] = (df['pred_minprobs']*10).round(0)/10  # 四舍五入到最近的0.1
        df_grouped = df.groupby('a_bins')['repred'].apply(lambda x: (x.str.lower() == 'unknow').mean()).reset_index()
        sns.lineplot(x='a_bins', y='repred', data=df_grouped, marker=marker, markersize=10, markerfacecolor='none', markeredgewidth=1.0, linewidth=1.0,  markeredgecolor=color, label=label, color=color, ax=ax)
        ax.grid(True, linestyle='--', alpha=0.6)

def plot_prob_unk(dfa, dfb, dfc):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))  # 创建一个1行3列的图形窗口

    label_list = ['Unknow Expression on T_k', 'Unknow Expression on T_unk']
    color_list = ['darkgreen', 'FireBrick']
    marker_list = ['o','^']
    name=['TriviaQA','NQ','PopQA']

    for i,df in enumerate([dfa,dfb,dfc]):
        df1,df2=cut_k_unk(df)
        plot_proportion(df1, df2, axs[i], label_list, color_list, marker_list)
        axs[i].set_xlabel('Model Confidence',fontsize=14)
        axs[i].set_ylabel('Proportion of Unknow Expression',fontsize=14)
        # axs[i].set_title(f'({chr(97+i)}) {name[i]}')
        axs[i].text(0.5, -0.2, f'({chr(97+i)}) {name[i]}', size=16, ha="center", transform=axs[i].transAxes)
        axs[i].legend(loc='lower left',fontsize=14)

    plt.tight_layout()
    plt.savefig("prob_plot.pdf")
    plt.show()


def cut_k_unk(df:pd.DataFrame):
    df=cal_know_unknow(df)
    df=cal_probs(df)
    return df[df['know']==1],df[df['unknow']==1]
    
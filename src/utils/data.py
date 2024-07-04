import pandas as pd
import ast
def gen_test_data(csv_file,num=2000,random_state=1):
    df=pd.read_csv(csv_file,sep='\t',header=None,names=['q', 'a'])
    df['a']=df['a'].apply(lambda x:ast.literal_eval(x))
    return df.sample(num,random_state=random_state)

def cal_know_unknow(df:pd.DataFrame)->pd.DataFrame:
    df['know']=df.apply(lambda x:int(x['pred'].lower() in [i.lower() for i in x['a']]),axis=1)
    df['unknow']=df['know'].apply(lambda x:1-x)
    return df

def cal_right_wrong_unk(df:pd.DataFrame)->pd.DataFrame:
    df['re_right']=df.apply(lambda x:int(x['repred'].lower() in [i.lower() for i in x['a']]),axis=1)
    df['re_unk']=df['repred'].apply(lambda x:int(x.lower()=='unknow'))
    df['re_wrong']=df.apply(lambda x:1-int(x['re_right'] or x['re_unk']),axis=1)
    return df

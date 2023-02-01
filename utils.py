import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold


def var_categoricas_cross_tab(df,normalize='columns'):    
    '''
    Function used to check the distribution of variables
    categorical according to the dependent variable: 'attrited' or 'existing'.
    '''
    for i in df.columns:
        if((i != 'Attrition_Flag')&(df.dtypes[i] == type(object))):
            print('--------------------------------------------------')
            print(i)
            print('--------------------------------------------------')
            print(pd.crosstab(df.Attrition_Flag,df[i],normalize=normalize))

def diferenca_estat_entre_var_numericas(df1,df2,alfa=0.05):
    '''
    Function used to check which numeric variables
    have statistically different distribution between the groups
    'attrited' and 'existing'.
    '''
    var_estat_diferentes = []
    for i in df1.columns:
        if df1[i].dtype != type(object):
            print('-----------------------------------')
            print(i)
            print(stats.ttest_ind(df1[i].values, df2[i].values))
            if(stats.ttest_ind(df1[i].values, df2[i].values)[1] < alfa):
                var_estat_diferentes.append(i)
    return var_estat_diferentes


class my_dictionary(dict):
    '''
    Class used to increment the used dictionary
    in the function 'variveis_categoricas_estat_diferentes'
    ''' 
    def __init__(self):    
        self = dict()

    def add(self, key, value):
        self[key] = value
    

    
    
def variaveis_categoricas_estat_diferentes(df):
    '''    
    Function used to verify which categories, among the categorical variables,
    show statistically different distribution between the 'attrited' and 'existing' groups.
    '''
    dicionario = my_dictionary()
    for i in df.columns:
        categorias_significativas = []
        if((df[i].dtype == type(object))&(i != 'Attrition_Flag')):
            df_aux = pd.crosstab(df.Attrition_Flag,df[i])
            categoria = df_aux.loc['Attrited Customer'].index
            sucesso = df_aux.loc['Attrited Customer'].values
            total = df_aux.sum().values
            for j in range(0,len(df_aux.columns) - 1):
                for k in range(j+1,len(df_aux.columns)):
                    p_value = proportions_ztest([sucesso[j],sucesso[k]], [total[j],total[k]], value=0.05)[1]
                    if(p_value < 0.05):
                        categorias_significativas.append([categoria[j],categoria[k],p_value])
            dicionario.add(i, categorias_significativas)
    return dicionario



def training_algorithms(X_train,y_train,algorithms,cross_validation=10,scoring='f1'):
    '''
    Function used to train the algorithms.
    '''
    cv = StratifiedKFold(cross_validation)
    performance = []
    for i in algorithms:
        cross_val = cross_val_score(i, X_train, y_train, cv=cv,scoring=scoring)
        performance.append([i.fit(X_train,y_train),cross_val])
    return performance
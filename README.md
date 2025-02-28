import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('medical_examination.csv')
    return df

def add_overweight_column(df):
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = (df['bmi'] > 25).astype(int)
    return df

def normalize_data(df):
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
    return df

def draw_cat_plot(df):
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', height=5, aspect=1.5)
    fig.set_axis_labels('Categorical Features', 'Count')
    fig.set_titles('Cardio = {col_name}')
    return fig

def draw_heat_map(df):
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    corr = df_heat.corr()
    mask = corr.where(pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', mask=mask, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap')
    plt.show()

def main():
    df = load_data()
    df = add_overweight_column(df)
    df = normalize_data(df)
    draw_cat_plot(df)
    draw_heat_map(df)

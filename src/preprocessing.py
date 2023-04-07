import numpy as np
import plotly.graph_objects as go
from pydantic.utils import deep_update
import json
import geopandas as gpd
import os
from pandas import DataFrame
import pandas as pd


def load_results(name: str = 'df.json') -> tuple[DataFrame, DataFrame]:

    rename_dict = {
        'Nordeste': 'Northeast',
        'Norte': 'North',
        'Centro-oeste': 'Central-West',
        'Sul': 'South',
        'Sudeste': 'Southeast',
    }
    df = gpd.read_file('df.json', driver='GeoJSON').drop(columns='index')
    df_2018 = gpd.read_file('df_2018.json', driver='GeoJSON').drop(columns='index')
    df_counties = gpd.read_file(os.path.join('files', 'counties_geo.json'))
    df_states = gpd.read_file(os.path.join('files', 'states_geo.json'))
    df_counties = df_counties.set_index(keys=['SG_UF', 'NM_MUNICIPIO']).sort_values(by=['SG_UF', 'NM_MUNICIPIO'])

    def calculate_percentages(df):
        opponent = 'LULA' if any(df.columns.isin(['VOTOS_LULA'])) else 'HADDAD'
        df['PERCENTAGE_BOLSONARO'] = df['VOTOS_BOLSONARO'] / (df['VOTOS_BOLSONARO'] + df[f'VOTOS_{opponent}']) * 100
        df[f'PERCENTAGE_{opponent}'] = 100 - df['PERCENTAGE_BOLSONARO']
        return df

    df = calculate_percentages(df)
    df_2018 = calculate_percentages(df_2018)
    df['DELTA'] = df['PERCENTAGE_BOLSONARO'] - df_2018['PERCENTAGE_BOLSONARO']

    # Calculate change from last election per region
    delta_region = df[['VOTOS_BOLSONARO', 'VOTOS_LULA', 'NM_REGIAO']].groupby('NM_REGIAO').sum()
    delta_region['VOTOS_BOLSONARO_PERCENT'] = delta_region.VOTOS_BOLSONARO / delta_region.sum(axis=1) * 100
    delta_region = delta_region.drop(columns=['VOTOS_LULA'])

    delta_region_2018 = df_2018[['VOTOS_BOLSONARO', 'VOTOS_HADDAD', 'NM_REGIAO']].groupby('NM_REGIAO').sum()
    delta_region_2018['VOTOS_BOLSONARO_PERCENT_2018'] = delta_region_2018.VOTOS_BOLSONARO / delta_region_2018.sum(axis=1) * 100
    delta_region_2018 = delta_region_2018.drop(columns=['VOTOS_HADDAD'])
    delta_region_2018 = delta_region_2018.rename({'VOTOS_BOLSONARO': 'VOTOS_BOLSONARO_2018'}, axis=1)

    delta_region = delta_region.join(delta_region_2018)
    delta_region = delta_region.reindex(sorted(delta_region.columns), axis=1)
    delta_region['DELTA'] = delta_region.VOTOS_BOLSONARO_PERCENT - delta_region.VOTOS_BOLSONARO_PERCENT_2018

    df['QT_APTOS_CUMSUM_PERCENT'] = df.QT_APTOS.cumsum() / df.QT_APTOS.sum() * 100

    df = df.sort_values(by='QT_APTOS', ascending=False)
    df = df.reset_index(drop=True)

    # Votes per region
    df_regions = df.groupby('SG_UF').agg({
        'VOTOS_BOLSONARO': sum,
        'VOTOS_LULA': sum,
        'AREA_KM2': sum,
    })
    df_regions = df_states.merge(df_regions, left_on='SG_UF', right_index=True)
    columns = ['geometry', 'NM_REGIAO', 'VOTOS_BOLSONARO', 'VOTOS_LULA']
    df_regions = df_regions[columns].dissolve(by='NM_REGIAO', aggfunc=sum, as_index=False)
    df_regions['PERCENTAGE_LULA'] = df_regions['VOTOS_LULA'] / (df_regions['VOTOS_BOLSONARO'] + df_regions['VOTOS_LULA']) * 100
    df_regions['PERCENTAGE_BOLSONARO'] = 100 - df_regions['PERCENTAGE_LULA']

    df_regions = df_regions.merge(delta_region['DELTA'], left_on='NM_REGIAO', right_index=True)

    # Add data from 2018
    bolsonaro_2018 = df_2018.groupby('NM_REGIAO')['VOTOS_BOLSONARO'].sum().reset_index(drop=True)
    haddad_2018 = df_2018.groupby('NM_REGIAO')['VOTOS_HADDAD'].sum().reset_index(drop=True)
    df_regions['PERCENTAGE_TOTAL_BOLSONARO_2018'] = bolsonaro_2018 / (bolsonaro_2018 + haddad_2018).sum() * 100
    df_regions['PERCENTAGE_TOTAL_BOLSONARO'] = df_regions['VOTOS_BOLSONARO'] / (df_regions['VOTOS_BOLSONARO'] + df_regions['VOTOS_LULA']).sum() * 100

    # Rename regions to English
    df_regions['NM_REGIAO'] = df_regions['NM_REGIAO'].map(rename_dict)
    df['NM_REGIAO'] = df['NM_REGIAO'].map(rename_dict)

    return df, df_regions


def load_json(filename, additional_dict=None) -> dict:
    if additional_dict is None: additional_dict = dict()
    with open(filename, 'r') as file:
        data = json.load(file)
    additional_dict = deep_update(additional_dict, data)
    return additional_dict


def create_bars(df_regions, template_layout, colors):

    # Bar plots
    layout_bar1 = load_json('figs/tab1_bar1_layout.json', template_layout)
    layout_bar2 = load_json('figs/tab1_bar2_layout.json', template_layout)
    layout_bar3 = load_json('figs/tab1_bar3_layout.json', template_layout)
    layout_bar4 = load_json('figs/tab1_bar4_layout.json', template_layout)
    traces_bar1 = []
    traces_bar2 = []
    traces_bar3 = []
    traces_bar4 = []
    bar_data = load_json('figs/tab1_bar_data.json')

    # Bar 1 and 2
    for name in ['LULA', 'BOLSONARO']:
        traces_bar1.append(go.Bar(
            x=df_regions['NM_REGIAO'],
            y=df_regions[f'VOTOS_{name}'],
            text=df_regions[f'PERCENTAGE_{name}'].round(2).astype(str) + '%',
            name=name.capitalize(),
            marker_color=colors[name],
            **bar_data
        ))
        for _, df_ in df_regions.iterrows():
            traces_bar2.append(go.Bar(
                x=[name.capitalize()],
                y=[df_[f'VOTOS_{name}']],
                text=df_['NM_REGIAO'],
                name=df_['NM_REGIAO'],
                marker_color=colors[name],
                **bar_data
            ))

    # Bar 3 and 4
    traces_bar3.append(go.Bar(
        x=df_regions['NM_REGIAO'],
        y=df_regions[f'DELTA'],
        # text=df_regions[f'PERCENTAGE_BOLSONARO{name}'].round(2).astype(str) + '%',
        text=df_regions['NM_REGIAO'],
        marker_color=np.where(df_regions['DELTA'] > 0, colors['BOLSONARO'], colors['LULA']),
        **bar_data
    ))
    for name in ['_2018', '']:
        year = '2018' if name == '_2018' else '2022'

        for _, df_ in df_regions.iterrows():
            traces_bar4.append(go.Bar(
                x=[year],
                y=[df_[f'PERCENTAGE_TOTAL_BOLSONARO{name}']],
                text=df_['NM_REGIAO'],
                name=df_['NM_REGIAO'],
                marker_color=colors['BOLSONARO'],
                **bar_data
            ))

    bar1 = go.Figure(data=traces_bar1, layout=layout_bar1)
    bar2 = go.Figure(data=traces_bar2, layout=layout_bar2)
    bar3 = go.Figure(data=traces_bar3, layout=layout_bar3)
    bar4 = go.Figure(data=traces_bar4, layout=layout_bar4)

    bars_dict = {
        '2022 Results': {
            'Aggregated': bar2,
            'Expanded': bar1
        },
        '2022 vs 2018 Comparison': {
            'Aggregated': bar4,
            'Expanded': bar3
        }
    }

    # bars_dict['2022 vs 2018 Comparison']['Aggregated'].show()
    # bars_dict['2022 vs 2018 Comparison']['Expanded'].show()
    # bars_dict['2022 Results']['Aggregated'].show()
    # bars_dict['2022 Results']['Expanded'].show()
    return bars_dict


if __name__ == '__main__':
    load_results()

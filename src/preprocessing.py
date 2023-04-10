from itertools import repeat
import numpy as np
import plotly.graph_objects as go
from pydantic.utils import deep_update
import json
import geopandas as gpd
import os
from pandas import DataFrame
import pandas as pd


def load_results(
        name: str = 'df.json'
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:

    rename_dict = {
        'Nordeste': 'Northeast',
        'Norte': 'North',
        'Centro-oeste': 'Central-West',
        'Sul': 'South',
        'Sudeste': 'Southeast',
    }
    df = gpd.read_file('files/df.json', driver='GeoJSON')
    df_2018 = gpd.read_file('files/df_2018.json', driver='GeoJSON')

    df_zz = pd.read_csv('files/df_zz.csv')
    df_zz_2018 = pd.read_csv('files/df_zz_2018.csv')

    df_counties = gpd.read_file('files/counties_geo.json')
    state2region = pd.read_csv('files/state2region.csv')
    df_regions_geo = gpd.read_file(('files/regions_geo.json'))

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

    df = df.sort_values(by='QT_APTOS', ascending=False)
    df = df.reset_index(drop=True)

    df['QT_APTOS_CUMSUM_PERCENT'] = 100 - (df.QT_APTOS.cumsum() / df.QT_APTOS.sum() * 100)

    # Votes per region
    df_regions = df.groupby('SG_UF').agg({
        'VOTOS_BOLSONARO': sum,
        'VOTOS_LULA': sum,
        'AREA_KM2': sum,
    })
    df_regions = state2region.merge(df_regions, left_on='SG_UF', right_index=True)
    columns = ['NM_REGIAO', 'VOTOS_BOLSONARO', 'VOTOS_LULA']
    df_regions = df_regions[columns].groupby('NM_REGIAO').sum()
    df_regions['PERCENTAGE_LULA'] = df_regions['VOTOS_LULA'] / (df_regions['VOTOS_BOLSONARO'] + df_regions['VOTOS_LULA']) * 100
    df_regions['PERCENTAGE_BOLSONARO'] = 100 - df_regions['PERCENTAGE_LULA']

    df_regions = df_regions.merge(delta_region['DELTA'], left_on='NM_REGIAO', right_index=True)

    # Add data from 2018
    bolsonaro_2018 = df_2018.groupby('NM_REGIAO')['VOTOS_BOLSONARO'].sum()
    haddad_2018 = df_2018.groupby('NM_REGIAO')['VOTOS_HADDAD'].sum()
    df_regions['PERCENTAGE_TOTAL_BOLSONARO_2018'] = bolsonaro_2018 / (bolsonaro_2018 + haddad_2018).sum() * 100
    df_regions['PERCENTAGE_TOTAL_BOLSONARO'] = df_regions['VOTOS_BOLSONARO'] / (df_regions['VOTOS_BOLSONARO'] + df_regions['VOTOS_LULA']).sum() * 100

    # Add geometry
    df_regions = df_regions_geo.merge(df_regions, on='NM_REGIAO')

    # Rename regions to English
    df_regions['NM_REGIAO'] = df_regions['NM_REGIAO'].map(rename_dict)
    df['NM_REGIAO'] = df['NM_REGIAO'].map(rename_dict)

    df['NM_REGIAO_INT'] = df['NM_REGIAO'].map({
        "Southeast": 0,
        "South": 1,
        "Central-West": 2,
        "North": 3,
        "Northeast": 4,
    })
    return df, df_regions, df_zz, df_2018, df_zz_2018


def load_json(filename: str, additional_dict=None) -> dict:
    if additional_dict is None: additional_dict = dict()
    with open(filename, 'r') as file:
        data = json.load(file)
    additional_dict = deep_update(additional_dict, data)
    return additional_dict


def create_bars(df: DataFrame,
                df_regions: DataFrame,
                df_zz: DataFrame,
                df_2018: DataFrame,
                df_zz_2018: DataFrame,
                template_layout: dict,
                colors: dict) -> dict:

    # Bar plots
    layout_bar1 = load_json('figs/tab1_bar1_layout.json', template_layout)
    layout_bar2 = load_json('figs/tab1_bar2_layout.json', template_layout)
    layout_bar3 = load_json('figs/tab1_bar3_layout.json', template_layout)
    layout_bar4 = load_json('figs/tab1_bar4_layout.json', template_layout)
    traces_bar1 = []
    traces_bar2 = []
    traces_bar3 = []
    traces_bar4 = []
    bar1_data = load_json('figs/tab1_bar1_data.json')
    bar2_data = load_json('figs/tab1_bar2_data.json')
    bar3_data = load_json('figs/tab1_bar3_data.json')
    bar4_data = load_json('figs/tab1_bar4_data.json')

    all_votes = df_zz[['VOTOS_BOLSONARO', 'VOTOS_LULA']].sum().sum() + df[['VOTOS_LULA', 'VOTOS_BOLSONARO']].sum().sum()

    all_votes_2018 = df_zz_2018[['VOTOS_BOLSONARO', 'VOTOS_HADDAD']].sum().sum() + df_2018[['VOTOS_HADDAD', 'VOTOS_BOLSONARO']].sum().sum()

    # Bar 1 and 2
    for name in ['LULA', 'BOLSONARO']:
        customdata = np.array([
            [name.capitalize()] * df_regions.shape[0],
            df_regions[f'VOTOS_{name}'].tolist()
        ]).T

        traces_bar1.append(go.Bar(
            x=df_regions['NM_REGIAO'],
            y=df_regions[f'VOTOS_{name}'],
            text=df_regions[f'PERCENTAGE_{name}'].round(2).astype(str) + '%',
            name=name.capitalize(),
            marker_color=colors[name],
            customdata=customdata,
            **bar1_data
        ))

        for _, df_ in df_regions.iterrows():

            candidate_votes = df_regions[f'VOTOS_{name}'].sum() + df_zz[f'VOTOS_{name}'].sum()
            percent_votes = (df_regions[f'VOTOS_{name}'].sum() + df_zz[f'VOTOS_{name}'].sum()) / all_votes * 100

            customdata = np.array([
                percent_votes,
                candidate_votes
            ]).reshape(1, -1)

            traces_bar2.append(go.Bar(
                x=[name.capitalize()],
                y=[df_[f'VOTOS_{name}']],
                text=df_['NM_REGIAO'],
                name=df_['NM_REGIAO'],
                marker_color=colors[name],
                customdata=customdata,
                **bar2_data
            ))

    # Bar 3 and 4
    traces_bar3.append(go.Bar(
        x=df_regions['NM_REGIAO'],
        y=df_regions[f'DELTA'],
        text=df_regions['NM_REGIAO'],
        marker_color=np.where(df_regions['DELTA'] > 0, colors['BOLSONARO'], colors['LULA']),
        **bar3_data
    ))

    for name in ['_2018', '']:
        year = '2018' if name == '_2018' else '2022'

        if year == '2022':
            percent_votes = (df[f'VOTOS_BOLSONARO'].sum() + df_zz[f'VOTOS_BOLSONARO'].sum()) / all_votes * 100
        else:
            percent_votes = (df_2018[f'VOTOS_BOLSONARO'].sum() + df_zz_2018[f'VOTOS_BOLSONARO'].sum()) / all_votes_2018 * 100

        for _, df_ in df_regions.iterrows():
            traces_bar4.append(go.Bar(
                x=[year],
                y=[df_[f'PERCENTAGE_TOTAL_BOLSONARO{name}']],
                text=[df_['NM_REGIAO']],
                name=df_['NM_REGIAO'],
                marker_color=colors['BOLSONARO'],
                customdata=[[percent_votes] * df_.shape[0]],
                **bar4_data
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

    return bars_dict


def create_tab1_maps(df: DataFrame, df_regions: DataFrame, template_layout: dict) -> dict:

    tab1_map_results = load_json('figs/tab1_map_data_results.json')
    tab1_map_delta = load_json('figs/tab1_map_data_delta.json')
    map_borders = load_json('figs/tab1_map_data_borders.json')
    map_names = load_json('figs/tab1_map_data_names.json')
    tab1_map_layout1 = load_json("figs/tab1_map_layout1.json", template_layout)
    tab1_map_layout2 = load_json("figs/tab1_map_layout2.json", template_layout)
    tab1_map_results.update(dict(
        z=df['PERCENTAGE_BOLSONARO'].values,  # type: ignore
        geojson=df.geometry.__geo_interface__,
        locations=df.index,
        customdata=df[['VOTOS_LULA', 'VOTOS_BOLSONARO', 'NM_MUNICIPIO', 'PERCENTAGE_LULA', 'PERCENTAGE_BOLSONARO', 'NM_REGIAO']],
    ))

    # Regions borders
    map_borders.update(dict(
        geojson=df_regions.geometry.__geo_interface__,
        locations=df_regions.index,
        z=[1.0] * df_regions.shape[0],
        customdata=df_regions['NM_REGIAO'].values,
    ))

    tab1_map1 = go.Figure([
        tab1_map_results,
        map_borders,
        map_names,
    ], layout=tab1_map_layout1)

    output = {
        'fig': tab1_map1,
        '2022 Results': tab1_map_layout1,
        '2022 vs 2018 Comparison': tab1_map_layout2,
        'borders': map_borders,
        'names': map_names,
    }

    return output


def create_tab2_cumsum(df: DataFrame, template_layout: dict) -> dict:

    cumsum_line = load_json('figs/tab2_cumsum_data_line.json')
    cumsum_filled = load_json('figs/tab2_cumsum_data_filled.json')
    cumsum_text = load_json('figs/tab2_cumsum_data_text.json')
    cumsum_layout = load_json('figs/tab2_cumsum_layout.json', template_layout)

    customdata = df[['NM_MUNICIPIO']]
    customdata.insert(1, 'QT_APTOS_CUMSUM_PERCENT', 100 - df['QT_APTOS_CUMSUM_PERCENT'])
    cumsum_line.update(dict(
        x=df.index + 1,
        y=df.QT_APTOS.cumsum(),
        customdata=customdata.reset_index().values,
    ))

    cumsum_colors = cumsum_filled.pop('colors')

    cumsum = go.Figure([
        cumsum_line,
        *[cumsum_filled.copy() for _ in range(5)],
        *[cumsum_text.copy() for _ in range(5)]],
        cumsum_layout)

    output = dict(
        fig=cumsum,
        text=cumsum_text,
        colors=cumsum_colors
    )

    return output


def create_tab2_map(df: DataFrame, borders: dict, template_layout: dict):
    map_results = load_json('figs/tab2_map_data_results.json')
    map_layout = load_json('figs/tab2_map_layout.json', template_layout)
    customdata = df[['NM_MUNICIPIO', 'QT_APTOS']].reset_index()
    customdata['index'] = customdata['index'] + 1
    map_results.update(dict(
        # z=df[:213]['QT_APTOS_CUMSUM_PERCENT'],
        z=df[:213]['NM_REGIAO_INT'],
        geojson=df[:213].geometry.__geo_interface__,
        locations=df[:213].index,
        customdata=customdata[:213].values,
    ))
    map = go.Figure([map_results, borders], map_layout)
    return map


if __name__ == '__main__':
    load_results()

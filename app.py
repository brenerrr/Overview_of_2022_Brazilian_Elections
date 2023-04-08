from collections import defaultdict
import dash_bootstrap_components as dbc
import sys
from dash import html, dcc, Input, Output, Patch, State, ctx
import dash
from plotly import graph_objects as go
import numpy as np
import plotly.io as pio
from src.preprocessing import *

pio.templates.default = "simple_white"

print("Starting")

df, df_regions = load_results()

rgba_red = 'rgba(178,24,43,1)'
rgba_blue = 'rgba(33,102,172,1)'
rgba_white = 'rgba(255,255,255,1)'
colors = dict(BOLSONARO=rgba_blue, LULA=rgba_red)
texts_html = load_json('text/text_snippets.json')

template_layout = load_json('figs/template.json')

try:
    with open('mapbox_token.txt', 'r') as f:
        token = f.read()
except:
    token = None

if token is None:
    template_layout['mapbox']['style'] = 'white-bg'
else:
    template_layout['mapbox']['accesstoken'] = token

# %% Figures

tab1_map = create_tab1_maps(df, df_regions, template_layout)

bars = create_bars(df_regions, template_layout, colors)

cumsum = create_tab2_cumsum(df, template_layout)

tab2_map = create_tab2_map(df, tab1_map['borders'], template_layout)


# %% App
config = {
    # 'modeBarButtonsToRemove': ['select', 'lasso2d', 'pan', 'zoom', 'autoScale', 'zoomIn', 'zoomOut'],
    'displaylogo': False}
tabs_content = [

    # Tab 1
    html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
        html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
            html.Div(style={'flex': 1}, className='tab1-plot-background', children=[
                dcc.Graph(
                    figure=tab1_map['fig'],
                    id='tab1-map',
                    clear_on_unhover=True,
                    className='tab1-map',
                    style={'border-radius': '15px', 'background-color': 'white'},
                    config=config
                ),

                html.Div(
                    dcc.RadioItems(
                        ['2022 Results', '2022 vs 2018 Comparison'],
                        '2022 Results',
                        inline=True,
                        className='map-radio',
                        id='map-radio'
                    )
                )
            ]),


            html.Div(style={'flex': 1, 'flext-direction': 'row'}, children=[
                # html.Button('Change Plot Mode', id='bar-button', n_clicks=0),
                dcc.Graph(
                    figure=bars['2022 Results']['Aggregated'],
                    id='bar',
                    # animate=True,
                    # animation_options={'frame': {'redraw': False, }, 'transition': {'duration': 750, 'ease': 'cubic-in-out', }, },
                    config=config),

                html.Div(
                    dcc.RadioItems(
                        ['Aggregated', 'Expanded'],
                        'Aggregated',
                        inline=True,
                        className='radio',
                        id='bar-radio'
                    )
                )
            ]),

        ]),

        html.Div(id=("tab1-text"), className='tab1-text'),

        dcc.Interval(
            id='tab1-interval-component',
            interval=60 * 1000,  # in milliseconds
            n_intervals=0
        )
    ]),
    # Tab 2
    html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
        html.Div(style={'flex': 1}, children=[
            dcc.Graph(figure=cumsum, id='tab2-cumsum',),
        ]),

        html.Div(style={'flex': 1}, children=[
            dcc.Graph(figure=tab2_map, id='tab2-map',),
        ]),

        dcc.Interval(
            id='tab2-interval-component',
            interval=3 * 1000,  # in milliseconds
            n_intervals=1
        )

    ]),

]
tabs_name = ['tab-0', 'tab-1']
tabs = dict(zip(tabs_name, tabs_content))


# app = JupyterDash(__name/__)
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


app.layout = html.Div(className='main', children=[

    html.H1('Overview of Brazilian Elections'),

    html.Br(),

    dbc.Tabs(id='tabs', class_name='tabs', children=[
        dbc.Tab(id=tabs_name[0], class_name="tab-individual", active_tab_class_name='active-tab', active_label_class_name='active-tab-label', label='Who won?'),
        dbc.Tab(id=tabs_name[1], class_name="tab-individual", active_tab_class_name='active-tab', active_label_class_name='active-tab-label', label='How concentrated are voters?'),
    ]),

    html.Br(),

    html.Div(id='tab-content'),


])


@ app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
)
def update_tab(tab):
    return tabs[tab]

# ****************************** Tab 1 ******************************


@ app.callback(
    Output('tab1-text', 'children'),
    Input('tab1-map', 'figure'),
    State('tab1-text', 'children')
)
def update_tab1_text(map, current_text):
    mask = np.array(map['data'][1]['z'], dtype=bool)
    if mask.any():
        region = df_regions['NM_REGIAO'][mask].values[0]
    else:
        region = 'None'

    text = texts_html.get('tab1-text', {}).get(region, '')
    if text == current_text:
        return dash.no_update
    else:
        return html.Div(text)


@app.callback(
    Output('bar', 'figure'),
    Input('bar-radio', 'value'),
    Input('map-radio', 'value'),
)
def toggle_tab1_bar(radio_bar, radio_map):
    fig = bars[radio_map][radio_bar]

    return fig


@app.callback(
    Output('bar', 'figure', allow_duplicate=True),
    Input('tab1-map', 'figure'),
    State('bar', 'figure'),
    State('bar-radio', 'value'),
    State('map-radio', 'value'),
    prevent_initial_call=True
)
def update_tab1_bar(map, bar, radio_bar, radio_map):

    mask = np.array(map['data'][1]['z'])
    # Candidates in x axis
    if radio_bar == 'Aggregated':
        patch = make_bar_aggregated(mask, radio_bar, bar, radio_map)

    # Regions in x axis
    else:
        patch = make_bar_expanded(mask, bar, radio_map)

    return patch


def make_bar_expanded(mask, bar, value_map):

    patch = Patch()

    # Make sure update is necessary
    current_values = []
    for trace in bar['data']:
        current_values.append(trace['y'])

    if value_map == '2022 Results':
        names = ['LULA', 'BOLSONARO']
        new_data = list(df_regions[f'VOTOS_{name}'] * mask for i, name in enumerate(names))
        indexes = list(range(len(new_data)))

    else:
        new_data = [df_regions['DELTA'] * mask]
        indexes = [0]

    if np.array_equiv(new_data, current_values): return dash.no_update
    # print('updating expanded bars')
    for i, val in zip(indexes, new_data):
        # print(i, val)
        patch['data'][i]['y'] = val
    patch['layout'] = bars[value_map]['Expanded'].layout

    print('\n\n')
    return patch


def make_bar_aggregated(mask, value, bar, value_map):
    patch = Patch()

    # Make sure update is necessary
    current_values = []
    for trace in bar['data']:
        current_values.append(trace['y'][0])

    if value_map == '2022 Results':
        new_values = (df_regions['VOTOS_LULA'] * mask).tolist() + (df_regions['VOTOS_BOLSONARO'] * mask).tolist()
        if new_values == current_values: return dash.no_update
    else:
        new_values = (df_regions['PERCENTAGE_TOTAL_BOLSONARO_2018'] * mask).tolist() + (df_regions['PERCENTAGE_TOTAL_BOLSONARO'] * mask).tolist()
        if np.array_equiv(new_values, current_values): return dash.no_update

    patch['layout'] = bars[value_map]['Aggregated']['layout']

    for i, value in enumerate(new_values):
        patch['data'][i]['y'] = [value]

    return patch


@app.callback(
    Output('tab1-map', 'figure', allow_duplicate=True),
    Input('map-radio', 'value'),
    prevent_initial_call=True
)
def toggle_tab1_map(option):
    if option == '2022 Results':
        z = df['PERCENTAGE_BOLSONARO'].values
    else:
        z = df['DELTA'].values

    layout = tab1_map[option]
    patch = Patch()
    patch['data'][0]['z'] = z
    patch['layout'] = layout

    return patch


@ app.callback(
    Output('tab1-map', 'figure'),
    Input('tab1-interval-component', 'n_intervals'),
    State('tab1-map', 'hoverData'),
    State('tab1-map', 'figure'),
)
def update_tab1_map(n, hoverData, map):

    patch_map = Patch()

    if hoverData:
        # Find region of click
        i = hoverData['points'][0]['pointNumber']
        region = map['data'][0]['customdata'][i][-1]
        mask = df_regions.NM_REGIAO == region
        new_z = mask.astype(float).values

        # Check if it is the same as the one currently selected
        current_z = map['data'][1]['z']
        same_region_previous = np.array_equiv(new_z, current_z)

        if not same_region_previous:
            regions_z = mask.astype(float).values
            patch_map['data'][1]['z'] = regions_z

    else:
        if not all(map['data'][1]['z']):
            regions_z = np.ones_like(map['data'][1]['z'])
            patch_map['data'][1]['z'] = regions_z

    return patch_map


# ****************************** Tab 2 ******************************


@ app.callback(
    Output('tab2-cumsum', 'figure'),
    Input('tab2-cumsum', 'hoverData'),
    State('tab2-cumsum', 'figure'),
)
def update_tab2_cumsum(hoverData, cumsum):
    # First call
    if not ctx.triggered_id: return draw_tab2_cumsum(1000, cumsum)

    # A call by interval component but there is no hover data
    if not hoverData: return dash.no_update

    # A call by interval component with hover data
    else:
        i = hoverData['points'][0]['pointNumber'] + 1
        return draw_tab2_cumsum(i, cumsum)


def draw_tab2_cumsum(i, cumsum):
    patch = Patch()

    df_ = df.loc[:i - 1, ['QT_APTOS', 'NM_REGIAO']]
    all_voters = df_['QT_APTOS'].sum()
    percent_voters = {}
    for region in ['South', 'Southeast', 'Central-West', 'North', 'Northeast']:
        percent_voters[region] = df_['QT_APTOS'][df_['NM_REGIAO'] == region].sum() / all_voters

    # Fill
    # df_ = df.iloc[:i + 1]
    x_fill = df_.index.values + 1
    # x_fill = np.concatenate((x_fill[0:1], x_fill[1::20], x_fill[-1:], ))
    # Add points for the rest of the curve
    last = df.shape[0]
    step = 10
    n_points = int((last - x_fill[-1]) / step)
    n_points = max(n_points, 1)
    x_fill = np.concatenate((
        x_fill[0:1],
        x_fill[1::20],
        np.linspace(x_fill[-1], last, n_points)),
    )

    y_fill = cumsum['data'][0]['y'][:i]
    # y_fill = np.concatenate((y_fill[0:1], y_fill[1::20], y_fill[-1:], ))

    y_fill = np.concatenate((
        y_fill[0:1],
        y_fill[1::20],
        np.linspace(y_fill[-1], y_fill[-1], n_points)),
    )

    iN = 0
    color = {
        'Northeast': '#493B2A',
        'North': '#593F62',
        'Central-West': '#7B6D8D',
        'South': '#8499B1',
        'Southeast': '#A5C4D4'
    }
    x_ = x_fill
    y_ = y_fill
    for j, (region, percent) in enumerate(percent_voters.items()):

        x_ = x_[iN:]
        y_ = y_[iN:]

        # If there are any counties from this region
        iN = (x_ <= (percent * x_[-1]) + x_[0]).sum() if x_.size > 0 else 0

        # Offset is necessary to avoid gap between filled areas
        offset = 1 if iN < x_.size else 0

        patch['data'][1 + j]['x'] = x_[:iN + offset]
        patch['data'][1 + j]['y'] = y_[:iN + offset]
        patch['data'][1 + j]['marker']['color'] = color[region]

        # # Text
        # x_text = x_[round(iN / 2)]
        # # x_text = max(x_text, 1000)
        # y_text = cumsum['data'][0]['y'][round(i / 2)] / 2
        # # y_text = max(25e6, y_text)

        # # Text
        # patch['data'][6 + j]['x'] = [x_text]
        # patch['data'][6 + j]['y'] = [y_text]

        # patch['data'][6 + j]['textfont']['size'] = np.interp(
        #     percent_voters[region],
        #     [0, 0.2],
        #     [0, tab2_cumsum_text['textfont']['size']]
        # )
        # patch['data'][6 + j]['text'] = "".join([
        #     # f"{df['QT_APTOS_CUMSUM_PERCENT'][i-1] : 0.3f}% of voters live in<br>",
        #     # f"{i / len(df) * 100 : 0.3f}% of counties",
        #     f"{percent_voters[region]*100 :.2f}%"
        # ])

    print('\n\n\n')

    return patch


@ app.callback(
    Output('tab2-map', 'figure'),
    Input('tab2-interval-component', 'n_intervals'),
    State('tab2-cumsum', 'hoverData'),
    State('tab2-map', 'figure')
)
def update_tab2_map(n, hoverData, map):

    # First call
    if not ctx.triggered_id: return draw_tab2_map(213, map)

    # A call by interval component but there is no hover data
    if not hoverData: return dash.no_update

    # A call by interval component with hover data
    else:
        i = hoverData['points'][0]['pointNumber'] + 1
        return draw_tab2_map(i, map)


def draw_tab2_map(i, map):
    z = np.ones(i)
    z = df['QT_APTOS_CUMSUM_PERCENT'][: i]

    # Do nothing if hover did not change
    if np.array_equiv(z, map['data'][0]['z']): return dash.no_update

    geojson = df[: i].geometry.__geo_interface__
    locations = df[: i].index
    customdata = df[: i][['NM_MUNICIPIO', 'QT_APTOS']].reset_index()
    customdata['index'] = customdata['index'] + 1
    customdata = customdata.values

    patch = Patch()

    patch['data'][0]['z'] = z
    patch['data'][0]['geojson'] = geojson
    patch['data'][0]['locations'] = locations
    patch['data'][0]['customdata'] = customdata

    return patch


if __name__ == "__main__":
    app.run_server(debug=True)
# %%

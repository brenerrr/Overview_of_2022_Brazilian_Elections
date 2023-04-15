# Overview of 2022 Brazilian Elections

This project consits of analyzing the electoral data from the 2022 Brazilian election. More specifically, data from each ballot is gathered and processed from official government sources. A few questions like “What changed from the previous election (2018)?” or “How spatially concentrated are voters?” are some of the questions I wished to answer.  In order to achieve this in a more compelling way, I created an interactive dashboard with Dash. You can check it out at https://brenerramos.com/overview-of-brazilian-elections/.


https://user-images.githubusercontent.com/36827826/231920684-6c9b1e87-09e0-4ff6-b12b-c08190ead5b7.mp4

## Installation

First create a conda environment, activate it and download all necessary modules

```
conda create -n env_dash python=3.10
conda activate env_dash
pip install -r requirements.txt
```

## Running the app with Python

Make sure to navigate to the `app/` folder and run the code from there.

```
cd app
python app.py
```

With the default inputs the app will be deployed on http://localhost.


## Running the app with Docker

You can also create and run a Docker image with the following commands (be sure to be in the root folder of the repository and not on the `app` folder).

```
docker build -t dashboard .
docker run -p 80:80 dashboard
```

With the default inputs the app will be deployed on http://localhost. If you change the default port on `app/inputs.json` make sure to change the command above as well as the Dockerfile.

## Inputs
Some parameters of the dashboard can be controlled with inputs stored on `app/inputs.json`


| Input                | Description                                               | Default Value        |
| -------------------- | --------------------------------------------------------- | -------------------- |
| tab1_update_interval | How frequent data in the first tab is updated in seconds  | 2                    |
| tab2_update_interval | How frequent data in the second tab is updated in seconds | 3                    |
| port                 | Port used to reach the Flask server                       | 80                   |
| host                 | The hostname to listen on                                 | "0.0.0.0"            |
| red                  | RGBA values for red color                                 | "rgba(178,24,43,1)"  |
| blue                 | RGBA values for red color                                 | "rgba(33,102,172,1)" |
| debug                | Run in debug mode                                         | false                |
| tab2_preload_n       | Index used to when loading data on second tab             | 213                  |
| logging_level        | Log level                                                 | "info"               |

## Mapbox token

In order to load the dashboard with a proper black background, you must have a mapbox token. For this, you need to create a free account at https://www.mapbox.com/. After that, you will have access to a default public token.

Create a simple text file `app/mapbox_token.txt` and paste the token code on it. When the app is executed, it will use that token to apply the proper syle on the map, resulting in a black background.

## Election Data

The data to create the app is already stored on `app/files/`. Those files can be generated with

```
cd app
python src/generate_dataframes.py
```

This will download the data from every ballot used in the election as well as the boundaries of all counties. Everything will be processed and then saved in the folder `app/files`. Not all files are necessary for running the dashboard but they are not deleted because they can be useful for other projects. The description of all files are displayed in the table below.

| Input                        | Description                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| df.json                      | GeoDataframe containing data from the 2022 election                                 |
| df.csv                       | Csv file containing data from the 2022 election without geometries (Excel friendly) |
| df_2018.json                 | GeoDataframe containing data from the 2018 election                                 |
| df.csv                       | Csv file containing data from the 2018 election without geometries (Excel friendly) |
| df_zz.csv                    | Csv file containing data from 2022 election from foreign countries                  |
| df_zz_2018.csv               | Csv file containing data from 2018 election from foreign countries                  |
| counties_geo.json            | GeoDataframe containing boundaries of all counties with simplified geometry         |
| regions_geo.json             | GeoDataframe containing boundaries of all regions with simplified geometry          |
| modelourna_numerointerno.csv | Csv file that relates ballots internal number to a model                            |
| state2region.csv             | Csv file mapping each state to a region                                             |

Probably the file that can be most useful for other projects is the `df.json` or `df.csv`. Below there is a description of their columns.

| Column            | Description                            |
| ----------------- | -------------------------------------- |
| SG_UF             | State abbreviation                     |
| NM_MUNICIPIO      | County name                            |
| AREA_KM2          | Spatial area in squared kilometers     |
| QT_APTOS          | Number of voters                       |
| QT_COMPARECIMENTO | Number of voters that showeded up      |
| QT_ABSTENCOES     | Number of voters that didn't show up   |
| VOTOS_BOLSONARO   | Number of votes for Bolsonaro          |
| VOTOS_LULA        | Number of votes for Lula               |
| VOTOS_B_N         | Number of white/null votes             |
| UE2009            | Number of ballots from model UE2009    |
| UE2010            | Number of ballots from model UE2010    |
| UE2011            | Number of ballots from model UE2011    |
| UE2013            | Number of ballots from model UE2013    |
| UE2015            | Number of ballots from model UE2015    |
| UE2020            | Number of ballots from model UE2020    |
| NM_REGIAO         | Region this county belongs to          |
| geometry          | Geometry of county (only on `df.json`) |



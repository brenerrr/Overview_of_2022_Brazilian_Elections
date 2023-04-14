# %% Import modules
import os
import pandas as pd
import requests
import zipfile
import shutil
import os
import geopandas as gpd
import numpy as np

# %% Download data
shutil.rmtree('./files', ignore_errors=True)
os.mkdir('./files')

states = [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB',
    'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO', 'ZZ']

# Election data
for state in states:
    BU_URL = f'https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2022/buweb/bweb_2t_{state}_311020221535.zip'
    print(f'\n Downloading data from {state}')
    print(BU_URL)
    req = requests.get(BU_URL)
    file_path = os.path.join('./files', f'{state}.zip')
    with open(file_path, 'wb') as output_file:
        output_file.write(req.content)

    # Extract csv file
    with zipfile.ZipFile(file_path, mode="r") as archive:
        files = [filename for filename in archive.namelist() if filename.endswith('.csv')]
        for file in files:
            archive.extract(file, './files')

    os.remove(file_path)

# Ballot models
URL = 'https://cdn.tse.jus.br/estatistica/sead/odsele/modelo_urna/modelourna_numerointerno.zip'
req = requests.get(URL)
file_path = os.path.join('./files', 'model_number.zip')
with open(file_path, 'wb') as output_file:
    output_file.write(req.content)

# Extract csv file
with zipfile.ZipFile(file_path, mode="r") as archive:
    files = [filename for filename in archive.namelist() if filename.endswith('.csv')]
    for file in files:
        archive.extract(file, './files')

os.remove(file_path)

# Geo data
URLS = [
    'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2021/Brasil/BR/BR_Municipios_2021.zip',
    'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2021/Brasil/BR/BR_UF_2021.zip'
]
print('Downloading maps')
for URL in URLS:
    req = requests.get(URL)
    filename = URL.split('/')[-1]
    file_path = os.path.join('./files', filename)
    with open(file_path, 'wb') as output_file:
        output_file.write(req.content)

    # Extract csv file
    with zipfile.ZipFile(file_path, mode="r") as archive:
        archive.extractall('./files')

    os.remove(file_path)

# 2018 election
directory_2018 = os.path.join('./files', '2018')
shutil.rmtree(directory_2018, ignore_errors=True)
os.mkdir(directory_2018)

states = [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB',
    'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO', 'ZZ']
# Election data
for state in states:
    BU_URL = f'https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2018/buweb/BWEB_2t_{state}_301020181744.zip'
    print(f'\n Downloading data from {state}')
    print(BU_URL)
    req = requests.get(BU_URL)
    file_path = os.path.join(directory_2018, f'{state}.zip')
    with open(file_path, 'wb') as output_file:
        output_file.write(req.content)

    # Extract csv file
    with zipfile.ZipFile(file_path, mode="r") as archive:
        files = [filename for filename in archive.namelist() if filename.endswith('.csv')]
        for file in files:
            archive.extract(file, directory_2018)

    os.remove(file_path)


# %% Simplify geo data
print('\nSimplifying geometries data')

gdf_states = gpd.read_file('./files/BR_UF_2021.shp')
gdf_states = gdf_states.rename({'SIGLA': 'SG_UF'}, axis=1)

# Save state to region mapping
gdf_states[['SG_UF', 'NM_REGIAO']].to_csv('./files/state2region.csv', index=False)

# Create region boundaries
gdf_regions = gdf_states[['NM_REGIAO', 'geometry']].dissolve(by='NM_REGIAO')
gdf_regions["geometry"] = (
    gdf_regions.to_crs(gdf_regions.estimate_utm_crs()).simplify(2000).to_crs(gdf_regions.crs)
)

with open('./files/regions_geo.json', 'w') as file:
    file.write(gdf_regions.reset_index().to_json(indent=2))


# Simplify counties geometry
gdf_counties = gpd.read_file('./files/BR_Municipios_2021.shp')
gdf_counties["geometry"] = (
    gdf_counties.to_crs(gdf_counties.estimate_utm_crs()).simplify(2000).to_crs(gdf_counties.crs)
)
gdf_counties = gdf_counties.rename({'NM_MUN': 'NM_MUNICIPIO', 'SIGLA': 'SG_UF'}, axis=1)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.upper()

# Fix names
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ESPIGÃO D'OESTE$", "ESPIGÃO DO OESTE", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ALVORADA D'OESTE$", "ALVORADA DO OESTE", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^LUÍS CORREIA$", "LUIS CORREIA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^AÇU$", "ASSÚ", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ARÊS$", "AREZ", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^JANUÁRIO CICCO$", "BOA SAÚDE", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^GRACHO CARDOSO$", "GRACCHO CARDOSO", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^CAÉM$", "CAEM", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^CAMACAN$", "CAMACÃ", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SANTO ESTÊVÃO$", "SANTO ESTEVÃO", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^OLHOS-D'ÁGUA$", "OLHOS D'ÁGUA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^PINGO-D'ÁGUA$", "PINGO D'ÁGUA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SEM-PEIXE$", "SEM PEIXE", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ANTÔNIO OLINTO$", "ANTONIO OLINTO", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SANTO ANTÔNIO DO CAIUÁ$", "SANTO ANTONIO DO CAIUÁ", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ELDORADO DO CARAJÁS$", "ELDORADO DOS CARAJÁS", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SANTA IZABEL DO PARÁ$", "SANTA ISABEL DO PARÁ", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^AMPARO DO SÃO FRANCISCO$", "AMPARO DE SÃO FRANCISCO", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^DONA EUZÉBIA$", "DONA EUSÉBIA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SÃO TOMÉ DAS LETRAS$", "SÃO THOMÉ DAS LETRAS", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^SÃO LUIZ DO PARAITINGA$", "SÃO LUÍS DO PARAITINGA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^WESTFÁLIA$", "WESTFALIA", regex=True)
gdf_counties.NM_MUNICIPIO = gdf_counties.NM_MUNICIPIO.str.replace(r"^ANHANGUERA$", "ANHANGÜERA", regex=True)
gdf_counties = gdf_counties.loc[~(gdf_counties.NM_MUNICIPIO == 'LAGOA MIRIM')]

with open('./files/counties_geo.json', 'w') as file:
    file.write(gdf_counties.to_json(indent=2))

gdf_counties = gdf_counties.set_index(keys=['SG_UF', 'NM_MUNICIPIO']).sort_values(by=['SG_UF', 'NM_MUNICIPIO'])
# %% Load election data
print('\nProcessing election data')

files = [file for file in os.listdir('./files') if file.startswith('bweb')]
df = []
for file in files:
    df_state = pd.read_csv(
        os.path.join('./files', file),
        encoding='Latin_1',
        delimiter=';',
        usecols=['DS_ELEICAO', 'SG_UF', 'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'QT_APTOS', 'QT_COMPARECIMENTO', 'QT_ABSTENCOES', 'NM_VOTAVEL', 'QT_VOTOS', 'NR_URNA_EFETIVADA'],
    )
    df_state = df_state[df_state.DS_ELEICAO == 'Eleição Geral Federal 2022'].drop(columns='DS_ELEICAO')
    df.append(df_state)

df = pd.concat(df, ignore_index=True)
df.head()


# %% Load ballots data

df_model_nr = pd.read_csv(os.path.join('./files', 'modelourna_numerointerno.csv'), encoding='Latin_1', delimiter=';', index_col='DS_MODELO_URNA')
nr_ballots = df[['SG_UF', 'NM_MUNICIPIO', 'NR_URNA_EFETIVADA']]
ballot_models_list = []

for model, values in df_model_nr.iterrows():
    s = (nr_ballots.NR_URNA_EFETIVADA >= values.NR_FAIXA_INICIAL) & (nr_ballots.NR_URNA_EFETIVADA <= values.NR_FAIXA_FINAL)
    s.rename(f'UE{model}', inplace=True)
    ballot_models_list.append(s.astype(int))

df_ballot_models = pd.concat(ballot_models_list, axis=1)
df_ballot_models = df_ballot_models.groupby(df_ballot_models.columns, axis=1).sum()
df_ballot_models = df_ballot_models.join(nr_ballots[['SG_UF', 'NM_MUNICIPIO']]).groupby(['SG_UF', 'NM_MUNICIPIO']).sum()
# df_ballot_models.head()

# %% Count votes

df_votes = df.groupby(['SG_UF', 'NM_MUNICIPIO', 'NM_VOTAVEL']).sum().pivot_table(columns='NM_VOTAVEL', index=['SG_UF', 'NM_MUNICIPIO'], values='QT_VOTOS')
df_votes['B_N'] = df_votes.Branco + df_votes.Nulo
df_votes.drop(columns=['Branco', 'Nulo'], inplace=True)
df_votes.rename({'B_N': 'VOTOS_B_N', 'JAIR BOLSONARO': 'VOTOS_BOLSONARO', 'LULA': 'VOTOS_LULA'}, inplace=True, axis=1)

# %% Get how many people could vote and how many showed up

df_condensed = df.groupby(['SG_UF', 'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO']).agg({
    'QT_APTOS': max,
    'QT_COMPARECIMENTO': max,
    'QT_ABSTENCOES': max,
})
df_condensed.reset_index(level=[2, 3], inplace=True, drop=True)
df_condensed = df_condensed.groupby(['SG_UF', 'NM_MUNICIPIO'], axis=0).sum()
# df_condensed.head()

# %% Merge data

df = df_condensed.join(df_votes).join(df_ballot_models)
df = gdf_counties[['geometry', 'AREA_KM2']].join(df, how='right')
df = df.reset_index()
df_zz = df[df.SG_UF == 'ZZ']
df = df[~(df.SG_UF == 'ZZ')]
df = df.merge(gdf_states[['SG_UF', 'NM_REGIAO']], how='left', on='SG_UF')
df = df.sort_values(by=['SG_UF', 'NM_REGIAO'])

# %% Export dataframe
df.to_file('./files/df.json', driver='GeoJSON', index=False)

# %% Export dataframe of international data

df_zz[['NM_MUNICIPIO', 'VOTOS_BOLSONARO', 'VOTOS_LULA']].to_csv('./files/df_zz.csv', index=False)
# %% Perform the same steps for 2018 data

directory_2018 = os.path.join('./files', '2018')
files = [file for file in os.listdir(directory_2018) if file.startswith('bweb')]
df_2018 = []
for file in files:
    df_state = pd.read_csv(
        os.path.join(directory_2018, file),
        encoding='Latin_1',
        delimiter=';',
        usecols=['DS_ELEICAO', 'SG_ UF', 'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO', 'QT_APTOS', 'QT_COMPARECIMENTO', 'QT_ABSTENCOES', 'NM_VOTAVEL', 'QT_VOTOS', 'NR_URNA_EFETIVADA'],
    )
    df_state = df_state[df_state.DS_ELEICAO == 'Eleição Geral Federal 2018'].drop(columns='DS_ELEICAO')
    df_2018.append(df_state)

df_2018 = pd.concat(df_2018, ignore_index=True)

# Small fixes
df_2018 = df_2018.rename({'SG_ UF': 'SG_UF'}, axis=1).sort_values(by=['SG_UF', 'NM_MUNICIPIO'])
df_2018.NM_MUNICIPIO = df_2018.NM_MUNICIPIO.str.replace(r"^QUINJINGUE$", "QUIJINGUE", regex=True)
df_2018.NM_MUNICIPIO = df_2018.NM_MUNICIPIO.str.replace(r"^ERERÊ$", "ERERÉ", regex=True)
df_2018.NM_MUNICIPIO = df_2018.NM_MUNICIPIO.str.replace(r"^ARÊS$", "AREZ", regex=True)
df_2018.NM_MUNICIPIO = df_2018.NM_MUNICIPIO.str.replace(r"^FORTALEZA DO TABOCÃO$", "TABOCÃO", regex=True)
df_2018 = df_2018[~(df_2018.NM_VOTAVEL == '#NULO#')]
i = df_2018[df_2018.NR_URNA_EFETIVADA == '#NULO#'].index
df_2018.loc[i, 'NR_URNA_EFETIVADA'] = 1295724
df_2018.NR_URNA_EFETIVADA = df_2018.NR_URNA_EFETIVADA.astype(np.int64)

#  Count df_votes
df_votes_2018 = df_2018.groupby(['SG_UF', 'NM_MUNICIPIO', 'NM_VOTAVEL']).sum().pivot_table(columns='NM_VOTAVEL', index=['SG_UF', 'NM_MUNICIPIO'], values='QT_VOTOS')
df_votes_2018['B_N'] = df_votes_2018.Branco + df_votes_2018.Nulo
df_votes_2018.drop(columns=['Branco', 'Nulo'], inplace=True)
df_votes_2018.rename({'B_N': 'VOTOS_B_N', 'JAIR BOLSONARO': 'VOTOS_BOLSONARO', 'FERNANDO HADDAD': 'VOTOS_HADDAD'}, inplace=True, axis=1)

# Get how many people could vote and how many showed up
df_condensed_2018 = df_2018.groupby(['SG_UF', 'NM_MUNICIPIO', 'NR_ZONA', 'NR_SECAO']).agg({
    'QT_APTOS': max,
    'QT_COMPARECIMENTO': max,
    'QT_ABSTENCOES': max,
})
df_condensed_2018.reset_index(level=[2, 3], inplace=True, drop=True)
df_condensed_2018 = df_condensed_2018.groupby(['SG_UF', 'NM_MUNICIPIO'], axis=0).sum()

# Find which ballot models were used
nr_ballots = df_2018[['SG_UF', 'NM_MUNICIPIO', 'NR_URNA_EFETIVADA']]
ballot_models_list = []

for model, values in df_model_nr.iterrows():
    s = (nr_ballots.NR_URNA_EFETIVADA >= values.NR_FAIXA_INICIAL) & (nr_ballots.NR_URNA_EFETIVADA <= values.NR_FAIXA_FINAL)
    s.rename(f'UE{model}', inplace=True)
    ballot_models_list.append(s.astype(int))

df_ballot_models_2018 = pd.concat(ballot_models_list, axis=1)
df_ballot_models_2018 = df_ballot_models_2018.groupby(df_ballot_models_2018.columns, axis=1).sum()
df_ballot_models_2018 = df_ballot_models_2018.join(nr_ballots[['SG_UF', 'NM_MUNICIPIO']]).groupby(['SG_UF', 'NM_MUNICIPIO']).sum()


df_final_2018 = df_condensed_2018.join(df_votes_2018).join(df_ballot_models_2018)

# Export zz data
df_final_2018.loc['ZZ', ['VOTOS_BOLSONARO', 'VOTOS_HADDAD']].to_csv('./files/df_zz_2018.csv')
# %%
df_final_2018 = gdf_counties[['geometry', 'AREA_KM2']].join(df_final_2018, how='right')
df_final_2018 = df_final_2018.reset_index()
df_zz_2018 = df_final_2018[df_final_2018.SG_UF == 'ZZ']
df_final_2018 = df_final_2018[~(df_final_2018.SG_UF == 'ZZ')]
df_final_2018 = df_final_2018.merge(gdf_states[['SG_UF', 'NM_REGIAO']], how='left', on='SG_UF')
df_final_2018 = df_final_2018.sort_values(by=['SG_UF', 'NM_REGIAO'])
df_final_2018.to_file('./files/df_2018.json', driver='GeoJSON', index=False)

# %% Delete unecessary files

for file in os.listdir('./files/'):
    if file.startswith('bweb') or file.startswith('BR_'):
        os.remove(f'./files/{file}')

shutil.rmtree('./files/2018')

print('FINISHED')

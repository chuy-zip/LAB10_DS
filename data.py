import pandas as pd


def get_time_series():
    
    consumo_2024 = pd.read_excel(
        './Datos_consumos/CONSUMO-HIDROCARBUROS-2024-12.xlsx',
        sheet_name='CONSUMO',
        header=6,
        parse_dates=['Fecha']
    )

    consumo_2025 = pd.read_excel(
        './Datos_consumos/VENTAS-HIDROCARBUROS-2025-05.xlsx',
        sheet_name='VENTAS_IMP',
        header=6,
        parse_dates=['Fecha']
    )

    consumo = pd.concat([consumo_2024, consumo_2025], ignore_index=True)
    consumo.sort_values('Fecha', inplace=True)

    consumo.set_index('Fecha', inplace=True)

    diesel_cols = [c for c in consumo.columns if 'Diesel' in c]
    consumo['Diesel'] = consumo[diesel_cols].sum(axis=1)

    df_consumo = consumo.rename(columns={
        'Gasolina regular': 'Gasolina Regular',
        'Gasolina superior': 'Gasolina Super',
        'Gas licuado de petróleo': 'Gas Licuado'
    })[['Gasolina Regular', 'Gasolina Super', 'Diesel', 'Gas Licuado']]

    df_consumo.head()
    df_consumo.index = pd.to_datetime(df_consumo.index, errors='coerce')

    df_consumo = df_consumo[df_consumo.index.notna()]

    df_consumo = df_consumo[~df_consumo.index.duplicated(keep='first')]

    df_consumo = df_consumo[df_consumo.index <= '2025-05-01']

    print(df_consumo.shape)  
    print("Desde", df_consumo.index.min(), "hasta", df_consumo.index.max())

    inicio = df_consumo.index.min()
    fin    = df_consumo.index.max()

    frecuencia = 'MS'

    df_consumo.index = pd.DatetimeIndex(df_consumo.index.values, freq=frecuencia)

    print(f"Inicio: {inicio.date()}")
    print(f"Fin:    {fin.date()}")
    print(f"Frecuencia: {df_consumo.index.freq}\n")

    return df_consumo 
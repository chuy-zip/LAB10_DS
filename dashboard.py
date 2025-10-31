import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from data import get_time_series

df_consumo = get_time_series()

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Combustible",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar paleta de colores
COLORES = {
    'azul_principal': '#3B74BF',
    'azul_oscuro': '#032F40',
    'azul_medio': '#013440',
    'verde_oscuro': '#488C35',
    'verde_claro': '#68BF2A'
}

# Título principal
st.title("Dashboard de Análisis de Combustible")
st.markdown("---")

# Sidebar para controles
with st.sidebar:
    st.header("Controles")
    st.markdown("---")
    
    # Selector de combustible
    combustible = st.selectbox(
        "Tipo de Combustible",
        ["Gasolina Super", "Diesel", "Todos"]
    )
    
    # Selector de modelo para predicciones
    modelo_seleccionado = st.selectbox(
        "Modelo de Predicción",
        ["Prophet", "Holt-Winters", "MLP", "Todos"]
    )
    
    # Rango de fechas
    fecha_inicio = st.date_input(
        "Fecha inicio",
        value=pd.to_datetime('2020-01-01')
    )
    fecha_fin = st.date_input(
        "Fecha fin",
        value=pd.to_datetime('2025-05-01')
    )

# Cargar datos (aquí cargarías tu DataFrame df_consumo)
# df_consumo = cargar_datos()

# VISUALIZACION 1: Serie temporal interactiva
st.header("Consumo Histórico")
st.markdown("Exploración del consumo de combustible a lo largo del tiempo")

if combustible == "Todos":
    fig1 = go.Figure()
    for col in ['Gasolina Super', 'Diesel']:
        fig1.add_trace(go.Scatter(
            x=df_consumo.index,
            y=df_consumo[col],
            name=col,
            line=dict(width=2)
        ))
else:
    fig1 = px.line(
        df_consumo, 
        x=df_consumo.index, 
        y=combustible,
        title=f"Consumo de {combustible}",
        color_discrete_sequence=[COLORES['azul_principal']]
    )

fig1.update_layout(
    hovermode='x unified',
    showlegend=True,
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

# VISUALIZACION 2: Comparación de modelos
st.header("Comparación de Modelos Predictivos")
st.markdown("Desempeño de diferentes modelos en los últimos 3 años")

# Datos de ejemplo para las métricas (debes reemplazar con tus métricas reales)
metricas_ejemplo = {
    'Modelo': ['Prophet', 'Holt-Winters', 'MLP'],
    'Gasolina Super RMSE': [57013, 176000, 70000],
    'Diesel RMSE': [77192, 222000, 137000]
}
df_metricas = pd.DataFrame(metricas_ejemplo)

# Gráfico de barras comparativas
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name='Gasolina Super',
    x=df_metricas['Modelo'],
    y=df_metricas['Gasolina Super RMSE'],
    marker_color=COLORES['azul_principal']
))
fig2.add_trace(go.Bar(
    name='Diesel',
    x=df_metricas['Modelo'],
    y=df_metricas['Diesel RMSE'],
    marker_color=COLORES['verde_oscuro']
))

fig2.update_layout(
    barmode='group',
    title="RMSE por Modelo y Combustible",
    height=400
)
st.plotly_chart(fig2, use_container_width=True)

# VISUALIZACION 3: Predicciones interactivas
st.header("Predicciones Futuras")
st.markdown("Proyección de consumo para los próximos meses")

col1, col2 = st.columns(2)

with col1:
    meses_prediccion = st.slider(
        "Meses a predecir",
        min_value=1,
        max_value=12,
        value=6
    )

with col2:
    combustible_pred = st.selectbox(
        "Combustible para predicción",
        ["Gasolina Super", "Diesel"],
        key="pred_combustible"
    )

# Aquí integrarías la lógica de predicción con los modelos cargados
# Por ahora mostramos un gráfico de ejemplo
fig3 = go.Figure()

# Datos históricos
fig3.add_trace(go.Scatter(
    x=df_consumo.index[-24:],  # Últimos 2 años
    y=df_consumo[combustible_pred][-24:],
    name='Histórico',
    line=dict(color=COLORES['azul_principal'], width=3)
))

# Predicción de ejemplo (reemplazar con predicción real)
futuro = pd.date_range(
    start=df_consumo.index[-1] + pd.DateOffset(months=1),
    periods=meses_prediccion,
    freq='MS'
)
prediccion_ejemplo = [df_consumo[combustible_pred].iloc[-1] * (1 + i*0.02) for i in range(meses_prediccion)]

fig3.add_trace(go.Scatter(
    x=futuro,
    y=prediccion_ejemplo,
    name='Predicción',
    line=dict(color=COLORES['verde_claro'], width=3, dash='dash')
))

fig3.update_layout(
    title=f"Predicción para {combustible_pred} - Próximos {meses_prediccion} meses",
    height=400
)
st.plotly_chart(fig3, use_container_width=True)

# VISUALIZACIÓN 4: Gráficos enlazados - Performance por período
st.header("Visualizaciones Enlazadas")
st.markdown("Comparación interactiva del desempeño de modelos")

# Selector de período para enlazar visualizaciones
periodo = st.selectbox(
    "Seleccionar período de análisis",
    ["Último año", "Últimos 3 años", "Período completo"],
    key="periodo_selector"
)

# Dos gráficos enlazados
col3, col4 = st.columns(2)

with col3:
    st.subheader("Error por Modelo")
    
    # Filtrar datos según período seleccionado
    if periodo == "Último año":
        datos_periodo = df_metricas.copy()  # Aquí usarías datos reales del período
    elif periodo == "Últimos 3 años":
        datos_periodo = df_metricas.copy()  # Datos de 3 años
    else:
        datos_periodo = df_metricas.copy()  # Todos los datos
    
    fig4a = px.bar(
        datos_periodo,
        x='Modelo',
        y=f'{combustible_pred} RMSE',
        color='Modelo',
        color_discrete_map={
            'Prophet': COLORES['azul_principal'],
            'Holt-Winters': COLORES['azul_oscuro'],
            'MLP': COLORES['verde_oscuro']
        }
    )
    st.plotly_chart(fig4a, use_container_width=True)

with col4:
    st.subheader("Tendencia de Error")
    
    # Gráfico de tendencia
    fig4b = go.Figure()
    modelos = ['Prophet', 'Holt-Winters', 'MLP']
    for modelo in modelos:
        fig4b.add_trace(go.Scatter(
            x=[1, 2, 3],  # Períodos de tiempo
            y=[50000, 45000, 40000],  # RMSE evolucionando
            name=modelo,
            mode='lines+markers'
        ))
    
    fig4b.update_layout(
        showlegend=True,
        height=300
    )
    st.plotly_chart(fig4b, use_container_width=True)

# Información adicional
st.markdown("---")
st.info("""
**Notas:** 
- Este dashboard muestra análisis exploratorio y predicciones de consumo de combustible
- Los modelos han sido entrenados con datos históricos desde 2000 hasta 2025
- Use los controles laterales para personalizar la visualización
""")
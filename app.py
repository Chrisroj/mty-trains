
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Configuración de página
st.set_page_config(layout="wide", page_title="Análisis de Fallas en Trenes")

st.title("📊🚉 Análisis de Fallas en Trenes MTY")

# Cargar datos
@st.cache_data
def load_data():

    # Se carga el archivo CSV limpio:
    #pth_file_clean = "./data/00_clean_data.csv"
    pth_file_clean = "./data/02_data_for_ML.csv"
    df = pd.read_csv(pth_file_clean, sep=';', encoding='utf-8')

    # Se ajustan los tipos de datos:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    categorical_cols = ['day_name', 'Veh', 'Linea', 'Sistema', 'Causó_desalojo', 'Supervisor_reviso', 'Cat', 'Fiabilidad_Servicio']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

df_clean = load_data()

# --- Filtros interactivos ---

st.sidebar.header("Filtros")

anios = st.sidebar.slider("Rango de Año", int(df_clean['year'].min()), int(df_clean['year'].max()), (int(df_clean['year'].min()), int(df_clean['year'].max())))
lineas = st.sidebar.multiselect("Selecciona Línea(s)", df_clean['Linea'].cat.categories, default=list(df_clean['Linea'].cat.categories))
categorias = st.sidebar.multiselect("Selecciona Categoría(s)", df_clean['Cat'].cat.categories, default=list(df_clean['Cat'].cat.categories))

with st.sidebar.expander("Selecciona Sistema(s)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Seleccionar todos", key="btn_seleccionar_sistemas"):
            st.session_state["sistemas"] = list(df_clean['Sistema'].cat.categories)
    
    with col2:
        if st.button("🧹 Limpiar valores", key="btn_limpiar_sistemas"):
            st.session_state["sistemas"] = []

    sistemas = st.multiselect(
        " ",
        options=df_clean['Sistema'].cat.categories,
        default=st.session_state.get("sistemas", list(df_clean['Sistema'].cat.categories)),
        label_visibility="collapsed",
        key="sistemas"
    )


with st.sidebar.expander("Selecciona Tren/Vehículo(s)", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Seleccionar todos", key="btn_seleccionar_vehiculos"):
            st.session_state["vehiculos"] = list(df_clean['Veh'].cat.categories)

    with col2:
        if st.button("🧹 Limpiar valores", key="btn_limpiar_vehiculos"):
            st.session_state["vehiculos"] = []

    vehiculos = st.multiselect(
        " ",
        options=df_clean['Veh'].cat.categories,
        default=st.session_state.get("vehiculos", list(df_clean['Veh'].cat.categories)),
        label_visibility="collapsed",
        key="vehiculos"
    )

# --- Asegurar que los filtros no estén vacíos ---
if not lineas:
    st.warning("Por favor, selecciona al menos una Línea.")
    st.stop()
if not sistemas:
    st.warning("Por favor, selecciona al menos un Sistema.")
    st.stop()
if not categorias:
    st.warning("Por favor, selecciona al menos una Categoría.")
    st.stop()
if not vehiculos:
    st.warning("Por favor, selecciona al menos un Tren/Vehículo.")
    st.stop()
if not anios:
    st.warning("Por favor, selecciona un rango de años.")
    st.stop()


# --- Aplicar filtros ---
df_filtered = df_clean[
    (df_clean['Linea'].isin(lineas)) &
    (df_clean['Sistema'].isin(sistemas)) &
    (df_clean['Cat'].isin(categorias)) &
    (df_clean['Veh'].isin(vehiculos)) &
    (df_clean['year'].between(anios[0], anios[1]))
]

# --- Gráficos principales ---
tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7 = st.tabs(["𝄜 Dataset","📈 Tendencias", "📊⚠️Distribución Fallas", "🕘Tiempos de retraso", "↩ Desalojos", "🧮 Analíticos", "🤖 Predicciones-ML"])

with tab_1:
    st.subheader("Dataset limpio")
    st.dataframe(df_filtered, use_container_width=True)

    #st.subheader("Descripción del Dataset")
    #st.write(df_filtered.describe(include='all'))

    #st.subheader("Información del Dataset")
    #buffer = df_filtered.info(buf=None)
    #st.text(buffer)


with tab_2:
    # ============== Gráficos de tendencias (semanal) ==============
    st.subheader("Fallas semanales por línea")

    fallas_semana = (
        df_filtered
        .assign(week=df_filtered['Fecha'].dt.isocalendar().week, year=df_filtered['Fecha'].dt.isocalendar().year)
        .groupby(["Linea", "year", "week"], observed=True)
        .size()
        .reset_index(name="conteo_fallas")
    )

    # Calculamos la fecha del primer día de cada semana (lunes)
    fallas_semana["Periodo"] = pd.to_datetime(
        fallas_semana["year"].astype(str) + "-W" + fallas_semana["week"].astype(str) + "-1",
        format="%G-W%V-%u"
    )

    # Gráfico de líneas por línea
    fig = px.line(
        fallas_semana, 
        x="Periodo", 
        y="conteo_fallas", 
        color="Linea", 
        #title="Fallas semanales por línea",
        labels={'Periodo': 'Año/Semana', 'conteo_fallas': 'Conteo de Fallas', 'Linea': 'Línea'}
    )

    # Agregar curva de estacionalidad (media móvil de 12 semanas por línea)
    for linea in fallas_semana['Linea'].cat.categories:
        datos_linea = fallas_semana[fallas_semana['Linea'] == linea].sort_values('Periodo')
        datos_linea['estacionalidad'] = datos_linea['conteo_fallas'].rolling(window=12, min_periods=1).mean()
        fig.add_scatter(
            x=datos_linea['Periodo'],
            y=datos_linea['estacionalidad'],
            mode='lines',
            name=f'Estacionalidad L{linea}',
            line=dict(dash='dash'),
            legendgroup=f'Estacionalidad L{linea}',
            showlegend=True
        )

    # Se muestra el gráfico:
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de tendencias (mensual) ==============
    st.subheader("Fallas mensuales por línea")
    
    # Agrupamos por año, mes y línea para visualizar tendencias
    fallas_mes = df_filtered.groupby(["Linea", "year", "month"], observed=True).size().reset_index(name="conteo_fallas")
    fallas_mes["Periodo"] = pd.to_datetime(fallas_mes[["year", "month"]].assign(day=1)) # <-- Se asigna el día 1 de cada mes para crear una fecha

    # Gráfico de líneas por línea
    fig = px.line(
        fallas_mes, 
        x="Periodo", 
        y="conteo_fallas", 
        color="Linea", 
        #title="Fallas mensuales por línea",
        labels={'Periodo': 'Año/Mes', 'conteo_fallas': 'Conteo de Fallas', 'Linea': 'Línea'}
    )

    # Agregar curva de estacionalidad (media móvil de 12 meses por línea)
    for linea in fallas_mes['Linea'].cat.categories:
        datos_linea = fallas_mes[fallas_mes['Linea'] == linea].sort_values('Periodo')
        datos_linea['estacionalidad'] = datos_linea['conteo_fallas'].rolling(window=12, min_periods=1).mean()
        fig.add_scatter(
            x=datos_linea['Periodo'],
            y=datos_linea['estacionalidad'],
            mode='lines',
            name=f'Estacionalidad L{linea}',
            line=dict(dash='dash'),
            legendgroup=f'Estacionalidad L{linea}',
            showlegend=True
        )

    st.plotly_chart(fig, use_container_width=True)

with tab_3:

    # ============== Gráficos de distribución de fallas por semana ==============
    st.subheader("Distribución de fallas por día de la semana")

    # Agrupamos por día y línea
    fallas_dia = df_filtered.groupby(["Linea", "day_name"], observed=True).size().reset_index(name="conteo_fallas")

    # Definimos el orden de los días de la semana y de las líneas
    orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Aplicamos el orden a las columnas categóricas
    fallas_dia['day_name'] = pd.Categorical(fallas_dia['day_name'], categories=orden_dias, ordered=True)

    # Gráfico de barras agrupadas respetando el orden
    fig = px.bar(
        fallas_dia.sort_values(['day_name', 'Linea']),
        x="day_name",
        y="conteo_fallas",
        color="Linea",
        barmode="group",
        #title="Fallas por día de la semana",
        labels={'conteo_fallas': 'Conteo de Fallas', 'day_name': 'Día', 'Linea': 'Línea'}    
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de distribución de fallas por categoría ==============
    st.subheader("Fallas por categoría")
    # Conteo de fallas por categoría y línea
    cat_failure = df_filtered.groupby(['Linea', 'Cat'], observed=True).size().reset_index(name="conteo_fallas")

    # Gráfico de barras por categoría
    fig = px.bar(
        cat_failure.sort_values(['Cat', 'Linea']), 
        x="Cat", 
        y="conteo_fallas", 
        color="Linea", 
        barmode="group", 
        #title="Fallas por categoría",
        labels={'conteo_fallas': 'Conteo de Fallas', 'Cat': 'Categoría', 'Linea': 'Línea'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de distribución de fallas por sistema y línea ==============
    st.subheader("Fallas por sistema y por línea")
    # Top 20 sistemas con más fallas, ordenados de mayor a menor en el eje y
    fallos_sistema = df_filtered.groupby(["Linea","Sistema"], observed=True).size().reset_index(name="conteo_fallas")

    fig = px.bar(
        fallos_sistema.sort_values(['Sistema', 'Linea']),
        y="conteo_fallas",
        x="Sistema",
        color= "Linea",
        orientation='v',
        #title="Fallas por Sistema y por Línea",
        labels={'conteo_fallas': 'Conteo de Fallas', 'Sistema': 'Sistema', 'Linea': 'Línea'}
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de distribución de fallas por tren y categoría ==============
    st.subheader("Fallas por tren y categoría")

    # Agrupamos por tren y categoría
    top_trenes = df_filtered.groupby(['Veh', 'Cat'], observed=True).size().reset_index(name="conteo_fallas")

    # Se convierte a "str" para que no haya errores en el gráfico
    #top_trenes['Veh'] = top_trenes['Veh'].astype(str)

    # Gráfico de barras por tren
    fig = px.bar(
        top_trenes.sort_values(['Veh','Cat']), 
        x="Veh", 
        y="conteo_fallas", 
        color="Cat", 
        #title="Fallas por tren y categoría",
        labels={'conteo_fallas': 'Conteo de Fallas', 'Veh': 'Id Tren', 'Cat': 'Categoría'}
        )
    fig.update_layout(xaxis={'categoryorder': 'total descending', 'tickangle': -45})
    st.plotly_chart(fig, use_container_width=True)


with tab_4:
    
    # ============== Gráficos de retraso promedio por sistema y línea ==============
    st.subheader("Retraso promedio (en minutos) por sistema y por línea")
    # Top 20 sistemas con más fallas, ordenados de mayor a menor en el eje y
    promedio_retraso_sistemas = (
        df_filtered.groupby(['Sistema', 'Linea'], observed=True)['Retraso_minutos'].mean()
        .reset_index(name='retraso_promedio')
        .sort_values('retraso_promedio', ascending=False)
        #.head(20)
    )

    fig = px.bar(
        promedio_retraso_sistemas.sort_values(['Sistema', 'Linea']),
        y="retraso_promedio",
        x="Sistema",
        color='Linea',
        orientation='v',
        #title="Retraso promedio (en minutos) por Sistema y por Línea",
        labels={'retraso_promedio': 'Retraso promedio (en minutos)', 'Sistema': 'Sistema', 'Linea': 'Línea'}
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de retraso promedio por categoría y línea ==============
    st.subheader("Retraso promedio (en minutos) por categoría y por línea")
    cat_delay = (
        df_filtered.groupby(['Linea','Cat'], observed=True)['Retraso_minutos'].mean()
        .reset_index(name='retraso_promedio')
        .sort_values('retraso_promedio', ascending=False)
        #.head(20)
    )

    # Crear gráfico de barras agrupadas
    fig = px.bar(
        cat_delay.sort_values(['Cat', 'Linea']),
        x='Cat',
        y='retraso_promedio',
        color='Linea',
        #title='Retraso promedio (en minutos) por Categoría y por Línea',
        labels={'retraso_promedio': 'Retraso promedio (minutos)', 'Cat': 'Categoría', 'Linea': 'Línea'}
    )
    # Configurar modo agrupado
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de retraso promedio por tren y línea ==============
    st.subheader("Retraso promedio (en minutos) por tren y por línea")

    # Top 20 sistemas con más fallas, ordenados de mayor a menor en el eje y
    promedio_retraso_trenes = (
        df_filtered.groupby(['Linea','Veh'], observed=True)['Retraso_minutos'].mean()
        .reset_index(name='retraso_promedio')
        .sort_values('retraso_promedio', ascending=False)
        #.head(20)
    )

    # Convertir Veh a string para visualización clara
    #promedio_retraso_trenes['Veh'] = promedio_retraso_trenes['Veh'].astype(str)

    fig = px.bar(
        promedio_retraso_trenes.sort_values(['Veh', 'Linea']),
        y="retraso_promedio",
        x="Veh",
        color = "Linea",
        orientation='v',
        #title="Retraso promedio (en minutos) por Tren y por Línea",
        labels={'retraso_promedio': 'Retraso promedio (en minutos)', 'Veh': 'Id Tren', 'Linea': 'Línea'}
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_5:
    # Filtrar solo fallas que causaron desalojo
    desalojos = df_filtered[df_filtered['Causó_desalojo'] == 1]

    # ============== Gráficos de desalojo por categoría y línea ==============
    st.subheader("Desalojos por Sistema y por Línea")
    # Agrupar por sistema y contar
    sistemas_desalojo = (
        desalojos.groupby(['Linea','Sistema'], observed=True)
        .size()
        .reset_index(name='conteo_desalojos')
        .sort_values('conteo_desalojos', ascending=False)
        #.head(20)
    )

    fig = px.bar(
        sistemas_desalojo.sort_values(['Sistema', 'Linea']),
        y="conteo_desalojos",
        x="Sistema",
        color='Linea',
        orientation='v',
        #title="Desalojos por Sistema y por Línea",
        labels={'conteo_desalojos': 'Conteo desalojos', 'Sistema': 'Sistema', 'Linea': 'Línea'}
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de desalojo por categoría y línea ==============
    st.subheader("Desalojos por Categoría y por Línea")
    cat_desalojo = (
        desalojos.groupby(['Linea','Cat'], observed=True)
        .size()
        .reset_index(name='conteo_desalojos')
        .sort_values('conteo_desalojos', ascending=False)
        #.head(20)
    )

    # Crear gráfico de barras agrupadas
    fig = px.bar(
        cat_desalojo.sort_values(['Cat', 'Linea']),
        x='Cat',
        y='conteo_desalojos',
        color='Linea',
        #title='Desalojos por Categoría y por Línea',
        labels={'conteo_desalojos': 'Conteo desalojos', 'Cat': 'Categoría', 'Linea': 'Línea'}
    )
    # Configurar modo agrupado
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráficos de desalojo por tren y línea ==============
    st.subheader("Desalojos por Tren y por Línea")
    # Agrupar por Veh y contar
    trenes_desalojo = (
        desalojos.groupby(['Linea','Veh'], observed=True)
        .size()
        .reset_index(name='conteo_desalojos')
        .sort_values('conteo_desalojos', ascending=False)
        #.head(20)
    )

    # Se convierte a "str" para que no haya errores en el gráfico
    trenes_desalojo['Veh'] = trenes_desalojo['Veh'].astype(str)

    fig = px.bar(
        trenes_desalojo.sort_values(['Veh', 'Linea']),
        y="conteo_desalojos",
        x="Veh",
        color='Linea',
        orientation='v',
        #title="Desalojos por Tren y por Línea",
        labels={'conteo_desalojos': 'Conteo desalojos', 'Veh': 'Id Tren', 'Linea': 'Línea'}
    )
    fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_6:

    # ============== Gráficos de Retraso promedio por sistema y causalidad de desalojo ==============
    st.subheader("Retraso promedio por sistema y causalidad de desalojo")
    # Agrupamos por sistema y causalidad
    causalidad = (
        df_filtered.groupby(['Sistema', 'Causó_desalojo'], observed=True)
        .agg(retraso_promedio=('Retraso_minutos', 'mean'))
        .reset_index()
    )

    # Convertimos a string para visualización limpia
    #causalidad['Causó_desalojo'] = causalidad['Causó_desalojo'].astype(str)

    # Gráfico de líneas o puntos comparando los grupos
    fig = px.scatter(
        causalidad,
        x='Sistema',
        y='retraso_promedio',
        color='Causó_desalojo',
        #title='Retraso promedio por sistema y causalidad de desalojo',
        labels={
            'Sistema': 'Sistema',
            'retraso_promedio': 'Retraso promedio (minutos)',
            'Causó_desalojo': '¿Causó desalojo?'
        }
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # ============== Relación entre % de desalojo y minutos de retraso ==============
    st.subheader("Relación entre % de desalojo y minutos de retraso")
    if not df_filtered['Cat'].cat.ordered:
        df_filtered['Cat'] = df_filtered['Cat'].cat.as_ordered()
    fig = px.scatter(
        df_filtered, x='Porcentaje_desalojo', y='Retraso_minutos', color='Linea',
        size='Cat', opacity=0.6, trendline='ols',
        hover_data=['Veh', 'Sistema']
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============== Gráfico de correlación ==============
    st.subheader("Heatmap de correlaciones")
    num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
    corr_spearman = df_filtered[num_cols].corr(method="spearman") # <-- Correlación de Spearman
    fig = px.imshow(corr_spearman, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)


with tab_7:
    st.subheader("Analítica Predictiva")
    st.markdown("**Predicción de desalojo**")

    # Cargar el pipeline entrenado
    pipeline_path = './artifacts/pipeline_rf_model.pkl'
    loaded_clf = joblib.load(pipeline_path)

    # Formulario interactivo
    with st.form("formulario_prediccion"):
        st.markdown("Ingresa los valores para realizar la predicción:")

        col1, col2 = st.columns(2)
        with col1:
            fecha = st.date_input("Fecha del incidente", value=pd.to_datetime(f"{int(df_clean['year'].max())}-01-01"))
            year = fecha.year
            month = fecha.month
            day = fecha.day
            day_name = fecha.strftime("%A")  # Nombre del día en inglés

            linea = st.selectbox("Línea", list(df_clean['Linea'].unique()))

        with col2:
            sistema = st.selectbox("Sistema", list(df_clean['Sistema'].unique()))
            veh = st.selectbox("Tren/Vehículo", list(df_clean['Veh'].unique()))
            desc = st.text_area("Descripción del fallo", value=f"Falla en sistema {sistema} ...")

        submitted = st.form_submit_button("Realizar predicción")

    if not submitted:
        st.warning("Antes de realizar una predicción, verifica que los datos anteriores sean correctos!")
        st.stop()

    if submitted:
        # Construir el diccionario de entrada
        x_sample = {
            'year': year,
            'month': month,
            'day': day,
            'day_name': day_name,
            'Linea': linea,
            'Sistema': sistema,
            'Veh': veh,
            'long_desc': len(desc)
        }

        # Ejecutar predicción
        y_pred = loaded_clf.predict(pd.DataFrame([x_sample]))

        st.markdown(
            f"<div style='text-align:center; font-size:2rem; font-weight:bold; color:green;'>🔮 Habrá desalojo: <span style='color:#1f77b4'>{y_pred[0]}</span></div>",
            unsafe_allow_html=True
        )

        st.write("**Probabilidades de desalojo:**")
        y_proba = loaded_clf.predict_proba(pd.DataFrame([x_sample]))
        proba_df = pd.DataFrame(y_proba, columns=loaded_clf.classes_)
        st.dataframe(proba_df, use_container_width=True)
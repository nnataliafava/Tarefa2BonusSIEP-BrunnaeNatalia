#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAREFA 2: Análise de Dados Imobiliários com ANOVA e Regressão Linear
Dashboard Interativo - Ames Housing Dataset

Alunas:
Brunna Gavião Moraes - 231025341
Natália Fava Deschamps Rabêlo - 231025379

Universidade de Brasília (UnB)
Faculdade de Tecnologia – FT | Departamento de Engenharia de Produção – EPR
Professor: João Gabriel de Moraes Souza
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise Imobiliária - Ames Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2e75b6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    .interpretation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e75b6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho
st.markdown('<h1 class="main-header">🏠 Análise Imobiliária - Ames Housing Dataset</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dashboard Interativo para Análise de Precificação Imobiliária</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">👥 Brunna Gavião Moraes & Natália Fava Deschamps Rabêlo | 📚 Prof. João Gabriel de Moraes Souza</p>', unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega e processa o dataset Ames Housing"""
    try:
        # Tenta carregar o arquivo
        base = pd.read_csv("AmesHousing.csv")
        
        # Mapeamento de colunas
        colunas = {
            'SalePrice': 'PrecoVenda',
            'Neighborhood': 'Bairro',
            'Exter Qual': 'QualidadeExterna',
            'Kitchen Qual': 'QualidadeCozinha',
            'Gr Liv Area': 'AreaConstruida',
            'Overall Qual': 'QualidadeGeral',
            'Year Built': 'AnoConstrucao'
        }
        
        # Filtra e renomeia colunas
        df = base[list(colunas.keys())].rename(columns=colunas).dropna()
        
        # Adiciona transformações logarítmicas
        df['Log_PrecoVenda'] = np.log(df['PrecoVenda'])
        df['Log_AreaConstruida'] = np.log(df['AreaConstruida'])
        
        return df, True
        
    except FileNotFoundError:
        st.error("❌ Arquivo 'AmesHousing.csv' não encontrado!")
        st.info("📁 Por favor, faça o upload do arquivo AmesHousing.csv")
        return None, False

# Carregamento dos dados
df, dados_carregados = load_data()

if not dados_carregados:
    # Interface para upload do arquivo
    uploaded_file = st.file_uploader("Faça o upload do arquivo AmesHousing.csv", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        colunas = {
            'SalePrice': 'PrecoVenda',
            'Neighborhood': 'Bairro',
            'Exter Qual': 'QualidadeExterna',
            'Kitchen Qual': 'QualidadeCozinha',
            'Gr Liv Area': 'AreaConstruida',
            'Overall Qual': 'QualidadeGeral',
            'Year Built': 'AnoConstrucao'
        }
        df = df[list(colunas.keys())].rename(columns=colunas).dropna()
        df['Log_PrecoVenda'] = np.log(df['PrecoVenda'])
        df['Log_AreaConstruida'] = np.log(df['AreaConstruida'])
        dados_carregados = True

if dados_carregados:
    # Sidebar com informações do dataset
    st.sidebar.header("📊 Informações do Dataset")
    st.sidebar.metric("Total de Observações", f"{len(df):,}")
    st.sidebar.metric("Preço Médio", f"${df['PrecoVenda'].mean():,.0f}")
    st.sidebar.metric("Variáveis Analisadas", "7")
    
    # Configurações da análise
    st.sidebar.header("⚙️ Configurações da Análise")
    alpha_nivel = st.sidebar.slider("Nível de Significância (α)", 0.01, 0.10, 0.05, 0.01)
    
    # Abas principais
    tab1, tab2, tab3 = st.tabs(["🔍 Análise Exploratória & ANOVA", "📈 Regressão Linear Múltipla", "📋 Conclusões & Recomendações"])
    
    # ========================================================================================
    # TAB 1: ANÁLISE EXPLORATÓRIA E ANOVA
    # ========================================================================================
    with tab1:
        st.header("🔍 Análise Exploratória e Comparativa com ANOVA")
        
        # Seleção de variáveis categóricas para análise
        st.subheader("Seleção de Variáveis Categóricas")
        variaveis_categoricas = ['Bairro', 'QualidadeExterna', 'QualidadeCozinha']
        var_selecionada = st.selectbox(
            "Escolha a variável categórica para análise:",
            variaveis_categoricas,
            help="Selecione uma variável categórica para realizar a análise ANOVA"
        )
        
        if var_selecionada:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Visualização com boxplot interativo
                fig = px.box(df, x=var_selecionada, y='PrecoVenda', 
                           title=f'Distribuição do Preço de Venda por {var_selecionada}')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Estatísticas descritivas
                st.subheader("📊 Estatísticas por Grupo")
                stats_grupo = df.groupby(var_selecionada)['PrecoVenda'].agg([
                    'count', 'mean', 'median', 'std'
                ]).round(2)
                stats_grupo.columns = ['N', 'Média', 'Mediana', 'Desvio Padrão']
                st.dataframe(stats_grupo)
        
        # Teste de pressupostos e ANOVA
        st.subheader("🧪 Teste de Hipóteses")
        
        if st.button("Executar Análise ANOVA", type="primary"):
            
            # Preparação dos dados para testes
            grupos = [grupo['PrecoVenda'].values for nome, grupo in df.groupby(var_selecionada)]
            
            # 1. Teste de Normalidade (Shapiro-Wilk por grupo)
            st.write("### 1. Teste de Normalidade (Shapiro-Wilk)")
            normalidade_resultados = []
            for nome, grupo in df.groupby(var_selecionada):
                if len(grupo) > 3:  # Shapiro-Wilk precisa de pelo menos 3 observações
                    stat, p_val = shapiro(grupo['PrecoVenda'])
                    normalidade_resultados.append({
                        'Grupo': nome,
                        'Estatística W': f"{stat:.4f}",
                        'p-valor': f"{p_val:.4f}",
                        'Normal?': "✅ Sim" if p_val > alpha_nivel else "❌ Não"
                    })
            
            if normalidade_resultados:
                st.dataframe(pd.DataFrame(normalidade_resultados))
            
            # 2. Teste de Homocedasticidade (Levene)
            st.write("### 2. Teste de Homocedasticidade (Levene)")
            levene_stat, levene_p = levene(*grupos)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estatística de Levene", f"{levene_stat:.4f}")
            with col2:
                st.metric("p-valor", f"{levene_p:.4f}")
            
            homoc_resultado = "✅ Variâncias homogêneas" if levene_p > alpha_nivel else "❌ Variâncias heterogêneas"
            st.write(f"**Resultado:** {homoc_resultado}")
            
            # 3. ANOVA Paramétrica
            st.write("### 3. ANOVA Paramétrica")
            modelo_anova = smf.ols(f'PrecoVenda ~ C({var_selecionada})', data=df).fit()
            anova_table = sm.stats.anova_lm(modelo_anova, typ=2)
            st.dataframe(anova_table)
            
            # 4. Teste de Kruskal-Wallis (Não-paramétrico)
            st.write("### 4. Teste de Kruskal-Wallis (Alternativa Não-paramétrica)")
            kruskal_stat, kruskal_p = kruskal(*grupos)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estatística H", f"{kruskal_stat:.4f}")
            with col2:
                st.metric("p-valor", f"{kruskal_p:.4e}")
            
            # 5. Teste Post-hoc (Tukey HSD)
            if kruskal_p < alpha_nivel:
                st.write("### 5. Teste Post-hoc: Tukey HSD")
                try:
                    tukey = pairwise_tukeyhsd(endog=df['PrecoVenda'], groups=df[var_selecionada], alpha=alpha_nivel)
                    st.text(str(tukey.summary()))
                except:
                    st.warning("Não foi possível executar o teste Tukey HSD para esta variável.")
            
            # Interpretação
            st.markdown("### 💡 Interpretação dos Resultados")
            
            if var_selecionada == 'Bairro':
                interpretacao = f"""
                <div class="interpretation-box">
                <h4>📍 Análise por Bairro</h4>
                <p><strong>p-valor Kruskal-Wallis:</strong> {kruskal_p:.4e}</p>
                <p><strong>Conclusão:</strong> {'Existe diferença significativa' if kruskal_p < alpha_nivel else 'Não há diferença significativa'} 
                nos preços medianos entre diferentes bairros (α = {alpha_nivel}).</p>
                <p><strong>Implicação Prática:</strong> A localização é um fator crucial na precificação imobiliária. 
                Investidores e corretores devem considerar o bairro como variável primária na avaliação de imóveis.</p>
                </div>
                """
            elif var_selecionada == 'QualidadeExterna':
                interpretacao = f"""
                <div class="interpretation-box">
                <h4>🏗️ Análise por Qualidade Externa</h4>
                <p><strong>p-valor Kruskal-Wallis:</strong> {kruskal_p:.4e}</p>
                <p><strong>Conclusão:</strong> {'Existe diferença significativa' if kruskal_p < alpha_nivel else 'Não há diferença significativa'} 
                nos preços medianos entre diferentes qualidades externas (α = {alpha_nivel}).</p>
                <p><strong>Implicação Prática:</strong> A qualidade do acabamento externo impacta diretamente o valor do imóvel. 
                Reformas externas podem ser um investimento com alto retorno.</p>
                </div>
                """
            else:  # QualidadeCozinha
                interpretacao = f"""
                <div class="interpretation-box">
                <h4>🍳 Análise por Qualidade da Cozinha</h4>
                <p><strong>p-valor Kruskal-Wallis:</strong> {kruskal_p:.4e}</p>
                <p><strong>Conclusão:</strong> {'Existe diferença significativa' if kruskal_p < alpha_nivel else 'Não há diferença significativa'} 
                nos preços medianos entre diferentes qualidades de cozinha (α = {alpha_nivel}).</p>
                <p><strong>Implicação Prática:</strong> A qualidade da cozinha é determinante no valor do imóvel. 
                Renovações na cozinha podem gerar valorização significativa.</p>
                </div>
                """
            
            st.markdown(interpretacao, unsafe_allow_html=True)
    
    # ========================================================================================
    # TAB 2: REGRESSÃO LINEAR MÚLTIPLA
    # ========================================================================================
    with tab2:
        st.header("📈 Modelagem Preditiva com Regressão Linear Múltipla")
        
        # Configuração do modelo
        st.subheader("⚙️ Configuração do Modelo")
        
        col1, col2 = st.columns(2)
        with col1:
            usar_log = st.checkbox("Usar transformação logarítmica", value=True, 
                                 help="Modelo log-log permite interpretação percentual dos coeficientes")
            incluir_constante = st.checkbox("Incluir intercepto", value=True)
        
        with col2:
            variaveis_numericas = ['AreaConstruida', 'QualidadeGeral', 'AnoConstrucao']
            vars_numericas_selecionadas = st.multiselect(
                "Variáveis numéricas:", 
                variaveis_numericas, 
                default=variaveis_numericas
            )
            
            variaveis_cat = ['QualidadeExterna', 'QualidadeCozinha']
            vars_cat_selecionadas = st.multiselect(
                "Variáveis categóricas:", 
                variaveis_cat, 
                default=variaveis_cat
            )
        
        if st.button("🚀 Executar Regressão Linear", type="primary"):
            
            # Preparação dos dados
            df_modelo = df.copy()
            
            if usar_log:
                y_var = 'Log_PrecoVenda'
                # Aplicar log nas variáveis numéricas se necessário
                for var in vars_numericas_selecionadas:
                    if var == 'AreaConstruida':
                        df_modelo[f'Log_{var}'] = df_modelo['Log_AreaConstruida']
                    else:
                        df_modelo[f'Log_{var}'] = np.log(df_modelo[var])
                vars_numericas_finais = [f'Log_{var}' if var == 'AreaConstruida' else f'Log_{var}' 
                                       for var in vars_numericas_selecionadas]
            else:
                y_var = 'PrecoVenda'
                vars_numericas_finais = vars_numericas_selecionadas
            
            # Criar variáveis dummy
            if vars_cat_selecionadas:
                dummies = pd.get_dummies(df_modelo[vars_cat_selecionadas], drop_first=True, dtype=float)
                df_modelo = pd.concat([df_modelo, dummies], axis=1)
                vars_dummy = list(dummies.columns)
            else:
                vars_dummy = []
            
            # Construir fórmula do modelo
            todas_vars = vars_numericas_finais + vars_dummy
            formula = f'{y_var} ~ ' + ' + '.join(todas_vars)
            
            # Ajustar modelo
            modelo = smf.ols(formula=formula, data=df_modelo).fit()
            
            # Exibir resultados
            st.subheader("📊 Resultados do Modelo")
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Ajustado", f"{modelo.rsquared_adj:.4f}")
            with col2:
                st.metric("R²", f"{modelo.rsquared:.4f}")
            with col3:
                st.metric("AIC", f"{modelo.aic:.1f}")
            with col4:
                st.metric("BIC", f"{modelo.bic:.1f}")
            
            # Tabela de coeficientes
            st.subheader("🔢 Coeficientes do Modelo")
            coef_df = pd.DataFrame({
                'Variável': modelo.params.index,
                'Coeficiente': modelo.params.values,
                'Erro Padrão': modelo.bse.values,
                't-valor': modelo.tvalues.values,
                'p-valor': modelo.pvalues.values,
                'IC 95% Inf': modelo.conf_int()[0].values,
                'IC 95% Sup': modelo.conf_int()[1].values
            })
            coef_df['Significativo'] = coef_df['p-valor'].apply(lambda x: '✅' if x < 0.05 else '❌')
            st.dataframe(coef_df.round(4))
            
            # Diagnósticos do modelo
            st.subheader("🩺 Diagnóstico dos Pressupostos")
            
            # 1. Gráficos de resíduos
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Resíduos vs Valores Ajustados', 'Q-Q Plot', 
                              'Histograma dos Resíduos', 'Resíduos Padronizados']
            )
            
            residuos = modelo.resid
            valores_ajustados = modelo.fittedvalues
            residuos_padronizados = (residuos - residuos.mean()) / residuos.std()
            
            # Resíduos vs Ajustados
            fig.add_trace(go.Scatter(x=valores_ajustados, y=residuos, mode='markers', name='Resíduos'),
                         row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Q-Q Plot
            from scipy.stats import probplot
            qq_data = probplot(residuos, dist="norm")
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q'),
                         row=1, col=2)
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0]*qq_data[0][0], 
                                   mode='lines', name='Linha Teórica'), row=1, col=2)
            
            # Histograma
            fig.add_trace(go.Histogram(x=residuos, name='Distribuição'), row=2, col=1)
            
            # Resíduos padronizados
            fig.add_trace(go.Scatter(x=valores_ajustados, y=residuos_padronizados, 
                                   mode='markers', name='Resíduos Padr.'), row=2, col=2)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(height=800, showlegend=False, title_text="Diagnósticos dos Resíduos")
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Testes estatísticos
            st.write("### Testes de Pressupostos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Teste de normalidade
                shapiro_stat, shapiro_p = shapiro(residuos)
                st.write("**Teste de Normalidade (Shapiro-Wilk):**")
                st.write(f"Estatística: {shapiro_stat:.4f}")
                st.write(f"p-valor: {shapiro_p:.4f}")
                st.write("✅ Resíduos normais" if shapiro_p > 0.05 else "❌ Resíduos não-normais")
            
            with col2:
                # Teste de heterocedasticidade
                try:
                    bp_stat, bp_p, _, _ = het_breuschpagan(residuos, modelo.model.exog)
                    st.write("**Teste de Heterocedasticidade (Breusch-Pagan):**")
                    st.write(f"Estatística: {bp_stat:.4f}")
                    st.write(f"p-valor: {bp_p:.4f}")
                    st.write("✅ Homocedasticidade" if bp_p > 0.05 else "❌ Heterocedasticidade")
                except:
                    st.write("❌ Não foi possível executar o teste Breusch-Pagan")
            
            # 3. VIF (Multicolinearidade)
            if len(todas_vars) > 1:
                st.write("### Teste de Multicolinearidade (VIF)")
                try:
                    X = df_modelo[todas_vars].dropna()
                    if incluir_constante:
                        X = sm.add_constant(X)
                    
                    vif_data = []
                    for i in range(X.shape[1]):
                        if X.columns[i] != 'const':
                            vif = variance_inflation_factor(X.values, i)
                            vif_data.append({'Variável': X.columns[i], 'VIF': vif})
                    
                    vif_df = pd.DataFrame(vif_data)
                    vif_df['Diagnóstico'] = vif_df['VIF'].apply(
                        lambda x: '✅ Baixo' if x < 5 else '⚠️ Moderado' if x < 10 else '❌ Alto'
                    )
                    st.dataframe(vif_df)
                except:
                    st.write("❌ Não foi possível calcular o VIF")
            
            # Interpretação dos coeficientes
            st.subheader("💡 Interpretação dos Coeficientes")
            
            interpretacao_coef = """
            <div class="interpretation-box">
            <h4>📊 Como Interpretar os Coeficientes</h4>
            """
            
            if usar_log:
                interpretacao_coef += """
                <p><strong>Modelo Log-Log:</strong> Os coeficientes representam elasticidades (variações percentuais).</p>
                <ul>
                """
                for idx, var in enumerate(coef_df['Variável']):
                    if var != 'Intercept' and coef_df.iloc[idx]['p-valor'] < 0.05:
                        coef_val = coef_df.iloc[idx]['Coeficiente']
                        if 'Log_' in var:
                            var_limpo = var.replace('Log_', '')
                            interpretacao_coef += f"<li><strong>{var_limpo}:</strong> +1% → +{coef_val*100:.2f}% no preço</li>"
                        else:
                            interpretacao_coef += f"<li><strong>{var}:</strong> Impacto categórico de {coef_val*100:.2f}%</li>"
                interpretacao_coef += "</ul>"
            else:
                interpretacao_coef += """
                <p><strong>Modelo Linear:</strong> Os coeficientes representam mudanças absolutas no preço.</p>
                """
            
            interpretacao_coef += "</div>"
            st.markdown(interpretacao_coef, unsafe_allow_html=True)
            
            # Recomendações práticas
            st.subheader("🎯 Recomendações Práticas")
            
            # Encontrar as variáveis mais impactantes
            coef_significativos = coef_df[
                (coef_df['p-valor'] < 0.05) & (coef_df['Variável'] != 'Intercept')
            ].copy()
            coef_significativos['Impacto_Abs'] = abs(coef_significativos['Coeficiente'])
            coef_significativos = coef_significativos.sort_values('Impacto_Abs', ascending=False)
            
            if not coef_significativos.empty:
                var_mais_impactante = coef_significativos.iloc[0]['Variável']
                coef_mais_impactante = coef_significativos.iloc[0]['Coeficiente']
                
                recomendacao = f"""
                <div class="interpretation-box">
                <h4>💼 Recomendações para o Mercado Imobiliário</h4>
                <p><strong>Fator Mais Impactante:</strong> {var_mais_impactante}</p>
                <p><strong>Para Vendedores:</strong> Foque em melhorar os atributos de maior impacto identificados pelo modelo.</p>
                <p><strong>Para Compradores:</strong> Considere estes fatores na negociação de preços.</p>
                <p><strong>Para Investidores:</strong> Priorize imóveis com potencial de melhoria nos fatores mais valorados.</p>
                </div>
                """
                st.markdown(recomendacao, unsafe_allow_html=True)
    
    # ========================================================================================
    # TAB 3: CONCLUSÕES E RECOMENDAÇÕES
    # ========================================================================================
    with tab3:
        st.header("📋 Conclusões e Recomendações Estratégicas")
        
        # Resumo executivo
        st.subheader("📈 Resumo Executivo")
        
        resumo = f"""
        <div class="interpretation-box">
        <h4>🎯 Principais Achados da Análise</h4>
        <p><strong>Dataset Analisado:</strong> {len(df):,} propriedades em Ames, Iowa</p>
        <p><strong>Preço Médio:</strong> ${df['PrecoVenda'].mean():,.0f}</p>
        <p><strong>Faixa de Preços:</strong> ${df['PrecoVenda'].min():,.0f} - ${df['PrecoVenda'].max():,.0f}</p>
        
        <h5>🏠 Fatores Mais Impactantes no Preço:</h5>
        <ol>
        <li><strong>Localização (Bairro):</strong> Diferenças significativas entre regiões</li>
        <li><strong>Qualidade Geral:</strong> Forte correlação com o valor do imóvel</li>
        <li><strong>Área Construída:</strong> Relação direta com o preço de venda</li>
        <li><strong>Qualidade dos Acabamentos:</strong> Impacto significativo na valorização</li>
        </ol>
        </div>
        """
        st.markdown(resumo, unsafe_allow_html=True)
        
        # Gráficos de correlação
        st.subheader("🔗 Matriz de Correlação das Variáveis Principais")
        
        corr_vars = ['PrecoVenda', 'AreaConstruida', 'QualidadeGeral', 'AnoConstrucao']
        corr_matrix = df[corr_vars].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Correlação entre Variáveis Numéricas")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuição de preços por quartis
        st.subheader("💰 Análise de Segmentação de Preços")
        
        df['Faixa_Preco'] = pd.qcut(df['PrecoVenda'], 4, labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Faixa_Preco', title="Distribuição por Faixa de Preço")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            faixa_stats = df.groupby('Faixa_Preco')['PrecoVenda'].agg(['min', 'max', 'mean']).round(0)
            faixa_stats.columns = ['Mínimo', 'Máximo', 'Média']
            st.write("**Estatísticas por Faixa:**")
            st.dataframe(faixa_stats)
        
        # Recomendações finais
        st.subheader("🚀 Recomendações Estratégicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="interpretation-box">
            <h4>👥 Para Corretores e Vendedores</h4>
            <ul>
            <li><strong>Enfatize a localização:</strong> Destaque as vantagens do bairro</li>
            <li><strong>Invista em qualidade:</strong> Pequenas melhorias podem gerar grande retorno</li>
            <li><strong>Precificação estratégica:</strong> Use o modelo para justificar preços</li>
            <li><strong>Marketing direcionado:</strong> Foque nos atributos mais valorizados</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="interpretation-box">
            <h4>💼 Para Investidores</h4>
            <ul>
            <li><strong>Oportunidades de reforma:</strong> Identifique imóveis com potencial</li>
            <li><strong>Análise de bairros:</strong> Invista em regiões promissoras</li>
            <li><strong>ROI planejado:</strong> Calcule retorno baseado nos fatores-chave</li>
            <li><strong>Diversificação:</strong> Considere diferentes faixas de preço</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Limitações e próximos passos
        st.subheader("⚠️ Limitações e Próximos Passos")
        
        limitacoes = """
        <div class="interpretation-box">
        <h4>🔍 Limitações do Estudo</h4>
        <ul>
        <li><strong>Escopo geográfico:</strong> Dados limitados a Ames, Iowa</li>
        <li><strong>Período temporal:</strong> Não considera variações temporais de mercado</li>
        <li><strong>Variáveis omitidas:</strong> Fatores como proximidade a amenidades não foram incluídos</li>
        <li><strong>Pressupostos estatísticos:</strong> Alguns pressupostos da regressão podem ser violados</li>
        </ul>
        
        <h4>🔮 Próximos Passos</h4>
        <ul>
        <li><strong>Modelos avançados:</strong> Implementar Random Forest ou Gradient Boosting</li>
        <li><strong>Variáveis adicionais:</strong> Incluir dados de localização e amenidades</li>
        <li><strong>Análise temporal:</strong> Considerar tendências de mercado ao longo do tempo</li>
        <li><strong>Validação externa:</strong> Testar modelo em outros mercados imobiliários</li>
        </ul>
        </div>
        """
        st.markdown(limitacoes, unsafe_allow_html=True)
        
        # Footer com créditos
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        📊 Dashboard desenvolvido por <strong>Brunna Gavião Moraes</strong> e <strong>Natália Fava Deschamps Rabêlo</strong><br>
        🎓 Universidade de Brasília (UnB) - Engenharia de Produção<br>
        👨‍🏫 Professor: João Gabriel de Moraes Souza
        </div>
        """, unsafe_allow_html=True)

else:
    st.stop()

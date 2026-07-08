import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

# ============================================================
# CONFIGURAÇÕES DE CAMINHO
# ============================================================
# Ajuste os nomes/pastas conforme onde os 3 scripts salvaram os arquivos
ARQUIVO_REFERENCIA = Path(__file__).parent / "Dados_Iraja_Prefeitura_Processado.csv"
ARQUIVO_SENSOR_1 = Path(__file__).parent / "Historico_50min_Processado.csv"
ARQUIVO_SENSOR_2 = Path(__file__).parent / "Airgradient_5min_Processado.csv"

ARQUIVO_SAIDA_RESUMO = Path(__file__).parent / "Estatisticas_Resumo.csv"
ARQUIVO_SAIDA_DETALHADO = Path(__file__).parent / "Dados_Alinhados_Detalhado.csv"
ARQUIVO_SAIDA_REGMULT_MODELO = Path(__file__).parent / "Regressao_Multipla_Modelo.csv"
ARQUIVO_SAIDA_REGMULT_COEF = Path(__file__).parent / "Regressao_Multipla_Coeficientes.csv"

# ============================================================
# 1. LEITURA DOS TRÊS ARQUIVOS JÁ PROCESSADOS (HORÁRIOS)
# ============================================================
for arq in [ARQUIVO_REFERENCIA, ARQUIVO_SENSOR_1, ARQUIVO_SENSOR_2]:
    if not os.path.exists(arq):
        raise FileNotFoundError(f"Arquivo não encontrado: {arq}")

df_ref = pd.read_csv(ARQUIVO_REFERENCIA)
df_s1 = pd.read_csv(ARQUIVO_SENSOR_1)
df_s2 = pd.read_csv(ARQUIVO_SENSOR_2)

# ============================================================
# 2. PADRONIZAÇÃO DA CHAVE DE TEMPO PARA O MERGE
# ============================================================
# Todos os scripts geram 'data_hora_cheia' no formato "%d/%m/%Y %H:%M:%S"
for df in [df_ref, df_s1, df_s2]:
    df['data_hora_cheia'] = pd.to_datetime(
        df['data_hora_cheia'], format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )

# ============================================================
# 3. RENOMEIA COLUNAS DE INTERESSE COM PREFIXO DE ORIGEM
#    (evita qualquer colisão e deixa claro no dado final)
# ============================================================
df_ref_sel = df_ref[['data_hora_cheia', 'ref_pm25', 'ref_pm10']].copy()

colunas_s1_disponiveis = [c for c in ['pm02', 'pm02_corrected', 'pm10', 'pm10_corrected', 'atmp', 'rhum'] if c in df_s1.columns]
df_s1_sel = df_s1[['data_hora_cheia'] + colunas_s1_disponiveis].copy()
df_s1_sel = df_s1_sel.rename(columns={
    'pm02': 'sensor1_pm25_raw',
    'pm02_corrected': 'sensor1_pm25_corr',
    'pm10': 'sensor1_pm10_raw',
    'pm10_corrected': 'sensor1_pm10_corr',
    'atmp': 'sensor1_temp',
    'rhum': 'sensor1_umid',
})

colunas_s2_disponiveis = [c for c in ['pm02', 'pm02_corrected', 'pm10', 'atmp', 'rhum'] if c in df_s2.columns]
df_s2_sel = df_s2[['data_hora_cheia'] + colunas_s2_disponiveis].copy()
df_s2_sel = df_s2_sel.rename(columns={
    'pm02': 'sensor2_pm25_raw',
    'pm02_corrected': 'sensor2_pm25_corr',
    'pm10': 'sensor2_pm10_raw',
    'atmp': 'sensor2_temp',
    'rhum': 'sensor2_umid',
})

# ============================================================
# 4. MERGE (ALINHAMENTO) PELAS HORAS EM COMUM
# ============================================================
df_alinhado = df_ref_sel.merge(df_s1_sel, on='data_hora_cheia', how='inner')
df_alinhado = df_alinhado.merge(df_s2_sel, on='data_hora_cheia', how='inner')

df_alinhado = df_alinhado.sort_values('data_hora_cheia').reset_index(drop=True)

print(f"Total de horas alinhadas entre os 3 conjuntos: {len(df_alinhado)}")

# ============================================================
# 5. DEFINIÇÃO DOS PARES (SENSOR x REFERÊNCIA) A COMPARAR
# ============================================================
# Cada item: (nome_sensor, versao, poluente, coluna_sensor, coluna_referencia)
pares_comparacao = [
    ('Sensor_1', 'Raw',        'PM2.5', 'sensor1_pm25_raw',  'ref_pm25'),
    ('Sensor_1', 'Corrigido',  'PM2.5', 'sensor1_pm25_corr', 'ref_pm25'),
    ('Sensor_1', 'Raw',        'PM10',  'sensor1_pm10_raw',  'ref_pm10'),
    ('Sensor_1', 'Corrigido',  'PM10',  'sensor1_pm10_corr', 'ref_pm10'),
    ('Sensor_2', 'Raw',        'PM2.5', 'sensor2_pm25_raw',  'ref_pm25'),
    ('Sensor_2', 'Corrigido',  'PM2.5', 'sensor2_pm25_corr', 'ref_pm25'),
    ('Sensor_2', 'Raw',        'PM10',  'sensor2_pm10_raw',  'ref_pm10'),
]

# ============================================================
# 6. FUNÇÃO DE CÁLCULO DAS MÉTRICAS PARA UM PAR
# ============================================================
def calcular_metricas(y_sensor, y_ref):
    """
    y_sensor: valores do sensor (eixo Y no scatter)
    y_ref:    valores da referência (eixo X no scatter, variável independente)
    """
    df_par = pd.DataFrame({'sensor': y_sensor, 'ref': y_ref}).dropna()

    n = len(df_par)
    if n < 3:
        return None  # dados insuficientes para regressão confiável

    x = df_par['ref'].values
    y = df_par['sensor'].values

    # Regressão linear: sensor = slope * referencia + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    erro = y - x
    rmse = np.sqrt(np.mean(erro ** 2))
    mae = np.mean(np.abs(erro))
    bias = np.mean(erro)  # positivo = sensor superestima; negativo = subestima

    return {
        'n_horas': n,
        'slope': round(slope, 4),
        'intercepto': round(intercept, 4),
        'r2': round(r_value ** 2, 4),
        'correlacao_pearson': round(r_value, 4),
        'p_value': round(p_value, 6),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'bias': round(bias, 4),
        'media_referencia': round(np.mean(x), 4),
        'media_sensor': round(np.mean(y), 4),
    }

# ============================================================
# 7. CALCULA PARA CADA PAR E MONTA A TABELA RESUMO
# ============================================================
linhas_resumo = []

for sensor, versao, poluente, col_sensor, col_ref in pares_comparacao:
    if col_sensor not in df_alinhado.columns:
        continue

    metricas = calcular_metricas(df_alinhado[col_sensor], df_alinhado[col_ref])

    if metricas is None:
        print(f"[AVISO] Dados insuficientes para {sensor} | {versao} | {poluente} — pulado.")
        continue

    linha = {
        'sensor': sensor,
        'versao': versao,
        'poluente': poluente,
    }
    linha.update(metricas)
    linhas_resumo.append(linha)

df_resumo = pd.DataFrame(linhas_resumo)

# ============================================================
# 8. REGRESSÃO LINEAR MÚLTIPLA
#    Referência = b0 + b1*Sensor_raw + b2*Umidade_sensor + b3*Temperatura_sensor
#    Objetivo: testar se umidade/temperatura melhoram a correção além do
#    que a regressão simples (só com o valor raw) já explica.
# ============================================================
def calcular_regressao_multipla(df, col_ref, col_sensor_raw, col_umid, col_temp):
    """
    Roda y = ref_pm ~ sensor_raw + umidade + temperatura via OLS (statsmodels).
    Retorna (stats_modelo, lista_coeficientes) ou (None, None) se dados insuficientes.
    """
    colunas_necessarias = [col_ref, col_sensor_raw, col_umid, col_temp]
    if not all(c in df.columns for c in colunas_necessarias):
        return None, None

    df_modelo = df[colunas_necessarias].dropna()
    n = len(df_modelo)

    # Regra prática: pelo menos 10 observações por variável preditora (3 preditores = 30 mínimo)
    if n < 30:
        return None, None

    y = df_modelo[col_ref]
    X = df_modelo[[col_sensor_raw, col_umid, col_temp]].rename(columns={
        col_sensor_raw: 'sensor_raw',
        col_umid: 'umidade',
        col_temp: 'temperatura',
    })
    X = sm.add_constant(X)  # adiciona o intercepto (b0)

    modelo = sm.OLS(y, X).fit()

    y_pred = modelo.predict(X)
    erro = y_pred - y
    rmse = np.sqrt(np.mean(erro ** 2))
    mae = np.mean(np.abs(erro))
    bias = np.mean(erro)

    stats_modelo = {
        'n_horas': n,
        'r2': round(modelo.rsquared, 4),
        'r2_ajustado': round(modelo.rsquared_adj, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'bias': round(bias, 4),
        'aic': round(modelo.aic, 2),
    }

    lista_coeficientes = []
    for nome_var in modelo.params.index:
        lista_coeficientes.append({
            'variavel': 'intercepto' if nome_var == 'const' else nome_var,
            'coeficiente': round(modelo.params[nome_var], 5),
            'erro_padrao': round(modelo.bse[nome_var], 5),
            'p_value': round(modelo.pvalues[nome_var], 6),
            'significativo_5pct': 'Sim' if modelo.pvalues[nome_var] < 0.05 else 'Não',
        })

    return stats_modelo, lista_coeficientes


# Um modelo por sensor, prevendo PM2.5 da referência a partir do valor raw + clima do próprio sensor
modelos_regressao_multipla = [
    ('Sensor_1', 'PM2.5', 'ref_pm25', 'sensor1_pm25_raw', 'sensor1_umid', 'sensor1_temp'),
    ('Sensor_2', 'PM2.5', 'ref_pm25', 'sensor2_pm25_raw', 'sensor2_umid', 'sensor2_temp'),
]

linhas_modelo_mult = []
linhas_coef_mult = []

for sensor, poluente, col_ref, col_raw, col_umid, col_temp in modelos_regressao_multipla:
    stats_modelo, coeficientes = calcular_regressao_multipla(
        df_alinhado, col_ref, col_raw, col_umid, col_temp
    )

    if stats_modelo is None:
        print(f"[AVISO] Dados insuficientes ou colunas ausentes para regressão múltipla de {sensor} — pulado.")
        continue

    linha_modelo = {'sensor': sensor, 'poluente': poluente}
    linha_modelo.update(stats_modelo)
    linhas_modelo_mult.append(linha_modelo)

    for coef in coeficientes:
        linha_coef = {'sensor': sensor, 'poluente': poluente}
        linha_coef.update(coef)
        linhas_coef_mult.append(linha_coef)

df_regmult_modelo = pd.DataFrame(linhas_modelo_mult)
df_regmult_coef = pd.DataFrame(linhas_coef_mult)

# ============================================================
# 9. SALVAMENTO DOS ARQUIVOS FINAIS PARA O STREAMLIT / POWER BI
# ============================================================
df_resumo.to_csv(ARQUIVO_SAIDA_RESUMO, index=False, sep=",", decimal=".", encoding="utf-8")

df_alinhado_saida = df_alinhado.copy()
df_alinhado_saida['data_hora_cheia'] = df_alinhado_saida['data_hora_cheia'].dt.strftime('%d/%m/%Y %H:%M:%S')
df_alinhado_saida.to_csv(ARQUIVO_SAIDA_DETALHADO, index=False, sep=",", decimal=".", encoding="utf-8")

df_regmult_modelo.to_csv(ARQUIVO_SAIDA_REGMULT_MODELO, index=False, sep=",", decimal=".", encoding="utf-8")
df_regmult_coef.to_csv(ARQUIVO_SAIDA_REGMULT_COEF, index=False, sep=",", decimal=".", encoding="utf-8")

print("=" * 60)
print("ESTATÍSTICAS DE COMPARAÇÃO CALCULADAS COM SUCESSO!")
print("=" * 60)
print(df_resumo.to_string(index=False))
print("-" * 60)
print("REGRESSÃO LINEAR MÚLTIPLA (Referência ~ Sensor_raw + Umidade + Temperatura)")
print("-" * 60)
if not df_regmult_modelo.empty:
    print(df_regmult_modelo.to_string(index=False))
    print()
    print(df_regmult_coef.to_string(index=False))
else:
    print("Nenhum modelo múltiplo pôde ser calculado (dados/colunas insuficientes).")
print("=" * 60)
print(f"Tabela resumo (regressão simples) salva em : {ARQUIVO_SAIDA_RESUMO}")
print(f"Tabela detalhada salva em                  : {ARQUIVO_SAIDA_DETALHADO}")
print(f"Regressão múltipla (modelo) salva em       : {ARQUIVO_SAIDA_REGMULT_MODELO}")
print(f"Regressão múltipla (coeficientes) salva em : {ARQUIVO_SAIDA_REGMULT_COEF}")
print("=" * 60)

"""
Relatório de Análise Comparativa de Sensores de Qualidade do Ar
LQA PUC-Rio — Sensor de Referência (Prefeitura) x Sensores Airgradient (3min / 5min)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Relatório - Sensores de Qualidade do Ar",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

REF_COLOR = "#2c3e50"   # Referência — cinza-chumbo
S1_COLOR = "#e67e22"    # Sensor 3min — laranja
S2_COLOR = "#2980b9"    # Sensor 5min — azul
RAW_STYLE = dict(dash="dot")
CORR_STYLE = dict(dash="solid")

VAR_ICONS = {"PM2.5": "💨", "PM10": "🌫️", "Temperatura": "🌡️", "Umidade": "💧"}
VAR_UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "Temperatura": "°C", "Umidade": "%"}

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    .filtro-box {background-color: rgba(31,111,92,0.06); padding: 0.9rem 1rem;
                 border-radius: 10px; border: 1px solid rgba(31,111,92,0.15); margin-bottom: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def parse_datetime(df, col="data_hora_cheia"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_csv(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    if "data_hora_cheia" in df.columns:
        df = parse_datetime(df)
    return df


@st.cache_data(show_spinner=False)
def build_merged(df_pref, df_3min, df_5min):
    """Une, hora a hora, referência + sensor 3min + sensor 5min num único dataframe.
    Temperatura/umidade usam apenas o valor bruto (o 'corrigido' do fabricante repete o bruto)."""
    parts = []
    if df_pref is not None:
        p = df_pref[["data_hora_cheia", "ref_pm25", "ref_pm10", "temperatura", "umidade_relativa"]].rename(
            columns={"temperatura": "ref_temp", "umidade_relativa": "ref_umid"})
        parts.append(p)
    if df_3min is not None:
        cols = {"pm02": "s1_pm25_raw", "pm02_corrected": "s1_pm25_corr", "pm10": "s1_pm10_raw",
                "pm10_corrected": "s1_pm10_corr", "atmp": "s1_temp", "rhum": "s1_umid"}
        avail = ["data_hora_cheia"] + [c for c in cols if c in df_3min.columns]
        parts.append(df_3min[avail].rename(columns=cols))
    if df_5min is not None:
        cols = {"pm02": "s2_pm25_raw", "pm02_corrected": "s2_pm25_corr", "pm10": "s2_pm10_raw",
                "atmp": "s2_temp", "rhum": "s2_umid"}
        avail = ["data_hora_cheia"] + [c for c in cols if c in df_5min.columns]
        parts.append(df_5min[avail].rename(columns=cols))

    if not parts:
        return None
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="data_hora_cheia", how="outer")
    return merged.sort_values("data_hora_cheia").reset_index(drop=True)


def get_comparison_columns(variavel, versao):
    """Retorna (col_ref, col_s1, col_s2, unidade) para a combinação escolhida."""
    if variavel in ("PM2.5", "PM10"):
        pol = "pm25" if variavel == "PM2.5" else "pm10"
        suf = "_raw" if versao == "Bruto" else "_corr"
        return f"ref_{pol}", f"s1_{pol}{suf}", f"s2_{pol}{suf}", VAR_UNITS[variavel]
    elif variavel == "Temperatura":
        return "ref_temp", "s1_temp", "s2_temp", VAR_UNITS[variavel]
    elif variavel == "Umidade":
        return "ref_umid", "s1_umid", "s2_umid", VAR_UNITS[variavel]


def linreg(x, y):
    """Regressão linear simples ignorando NaNs."""
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x, y = np.asarray(x[mask], dtype=float), np.asarray(y[mask], dtype=float)
    n = len(x)
    if n < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson_r = np.corrcoef(x, y)[0, 1] if n > 1 else np.nan
    rmse = np.sqrt(np.mean((y - x) ** 2))
    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    return dict(slope=slope, intercept=intercept, r2=r2, n=n, pearson_r=pearson_r,
                rmse=rmse, mae=mae, bias=bias)


def completeness_badge(pct):
    if pd.isna(pct):
        return "—"
    if pct >= 90:
        return f"🟢 {pct:.1f}%"
    elif pct >= 70:
        return f"🟡 {pct:.1f}%"
    else:
        return f"🔴 {pct:.1f}%"


def period_selector(df, key_prefix, help_text=None):
    """Filtro de período didático: intervalo contínuo OU dias específicos (não sequenciais)."""
    if df is None or df.empty or df["data_hora_cheia"].isna().all():
        return df, []

    st.markdown('<div class="filtro-box">', unsafe_allow_html=True)
    st.markdown("**🗓️ Escolha o período**" + (f" — {help_text}" if help_text else ""))
    modo = st.radio(
        "Modo de seleção",
        ["Intervalo contínuo", "Dias específicos"],
        horizontal=True, key=f"{key_prefix}_modo", label_visibility="collapsed",
        help="Intervalo contínuo: um período seguido (ex.: 01/06 a 10/06). "
             "Dias específicos: escolha e compare dias avulsos, mesmo que não sejam seguidos.",
    )
    dates_available = sorted(df["data_hora_cheia"].dt.date.dropna().unique())

    if modo == "Intervalo contínuo":
        d_range = st.date_input(
            "Intervalo", value=(dates_available[0], dates_available[-1]),
            min_value=dates_available[0], max_value=dates_available[-1],
            key=f"{key_prefix}_range", label_visibility="collapsed",
        )
        if isinstance(d_range, tuple) and len(d_range) == 2:
            start, end = d_range
            sel_dates = [d for d in dates_available if start <= d <= end]
        else:
            sel_dates = dates_available
    else:
        default = dates_available[-3:] if len(dates_available) >= 3 else dates_available
        sel_dates = st.multiselect(
            "Dias", dates_available, default=default,
            format_func=lambda d: d.strftime("%d/%m/%Y"),
            key=f"{key_prefix}_days", label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not sel_dates:
        st.info("Selecione ao menos um dia para ver os dados.")
        return df.iloc[0:0], []

    mask = df["data_hora_cheia"].dt.date.isin(sel_dates)
    return df[mask].sort_values("data_hora_cheia"), sel_dates


def variable_selector(key_prefix, include_versao=True):
    """Seletor didático de Variável (+ Versão bruto/corrigido quando aplicável)."""
    c1, c2 = st.columns([2, 2])
    with c1:
        variavel = st.radio(
            "🔬 Qual variável comparar?",
            ["PM2.5", "PM10", "Temperatura", "Umidade"],
            format_func=lambda v: f"{VAR_ICONS[v]} {v}",
            horizontal=True, key=f"{key_prefix}_var",
        )
    versao = "Bruto"
    with c2:
        if variavel in ("PM2.5", "PM10") and include_versao:
            versao = st.radio(
                "⚙️ Versão do sensor",
                ["Bruto", "Corrigido"],
                horizontal=True, key=f"{key_prefix}_versao",
                help="Bruto = leitura direta do sensor de baixo custo. "
                     "Corrigido = leitura ajustada por um modelo de calibração.",
            )
        else:
            st.caption("🌡️💧 Temperatura e umidade só têm uma versão: os sensores 1 e 2 reportam um valor "
                        "'corrigido' idêntico ao bruto, então usamos apenas o valor medido.")
    return variavel, versao


# ============================================================
# SIDEBAR — CARREGAMENTO DE DADOS
# ============================================================
st.sidebar.title("📁 Dados de entrada")
st.sidebar.caption("Envie os arquivos CSV exportados do pipeline de coleta.")

up_pref = st.sidebar.file_uploader("Sensor de Referência (Prefeitura) — horário", type="csv", key="up_pref")
up_3min = st.sidebar.file_uploader("Sensor Airgradient — 3 em 3 min", type="csv", key="up_3min")
up_5min = st.sidebar.file_uploader("Sensor Airgradient — 5 em 5 min", type="csv", key="up_5min")
up_stat = st.sidebar.file_uploader("Estatísticas Resumo (opcional)", type="csv", key="up_stat")

with st.sidebar.expander("ℹ️ Quais arquivos usar em cada campo?"):
    st.markdown(
        "- **Referência**: `Prefeitura.csv`\n"
        "- **3 em 3 min**: `Airgradient_3min.csv`\n"
        "- **5 em 5 min**: `Airgradient_5min.csv`\n"
        "- **Estatísticas** *(opcional)*: `Estatisticas_Resumo.csv` — usado só na última aba, "
        "como conferência dos números calculados no seu próprio pipeline.\n\n"
        "As comparações entre sensores são calculadas automaticamente aqui dentro — "
        "não é preciso enviar um arquivo já alinhado."
    )

df_pref = load_csv(up_pref)
df_3min = load_csv(up_3min)
df_5min = load_csv(up_5min)
df_stat = load_csv(up_stat)
df_merged = build_merged(df_pref, df_3min, df_5min)

any_loaded = any(d is not None for d in [df_pref, df_3min, df_5min])

st.title("🌫️ Relatório de Análise — Sensores de Qualidade do Ar")
st.caption("LQA PUC-Rio · Comparação entre sensor de referência e sensores de baixo custo (Airgradient)")

if not any_loaded:
    st.info("👈 Envie pelo menos um arquivo CSV na barra lateral para começar a análise.")
    st.markdown(
        """
        **Sobre este relatório**

        Este painel compara três fontes de dados, hora a hora:

        | Sensor | Frequência de medição | Papel |
        |---|---|---|
        | Prefeitura | 1 em 1 hora | Referência (padrão-ouro) |
        | Airgradient A | 3 em 3 minutos | Sensor de baixo custo, agregado por hora |
        | Airgradient B | 5 em 5 minutos | Sensor de baixo custo, agregado por hora |

        Para PM2.5 e PM10, cada sensor de baixo custo tem leitura **bruta** e **corrigida**.
        Para temperatura e umidade, usamos apenas o valor bruto (o "corrigido" do fabricante
        repete o mesmo número, então não agrega comparação real).
        """
    )
    st.stop()

# ============================================================
# TABS
# ============================================================
tab_overview, tab_ref, tab_s1, tab_s2, tab_compare, tab_stats = st.tabs(
    ["📊 Visão Geral", "🏛️ Referência (Prefeitura)", "🟠 Sensor 3min", "🔵 Sensor 5min",
     "📈 Comparação & Regressão", "📋 Estatísticas Resumo"]
)

# ------------------------------------------------------------
# TAB 1 — VISÃO GERAL
# ------------------------------------------------------------
with tab_overview:
    st.subheader("Visão Geral do Período")

    cols = st.columns(4)
    if df_pref is not None:
        cols[0].metric("Período (Referência)",
                        f"{df_pref['data_hora_cheia'].min().date()} → {df_pref['data_hora_cheia'].max().date()}")
        cols[0].caption(f"{len(df_pref)} horas registradas")
    if df_3min is not None:
        comp = (df_3min["medicoes_hora_completa"] >= 20).mean() * 100 if "medicoes_hora_completa" in df_3min else np.nan
        cols[1].metric("Sensor 3min — completude", completeness_badge(comp))
        cols[1].caption("% de horas com ≥20 medições (esperado)")
    if df_5min is not None:
        comp = (df_5min["medicoes_hora_completa"] >= 12).mean() * 100 if "medicoes_hora_completa" in df_5min else np.nan
        cols[2].metric("Sensor 5min — completude", completeness_badge(comp))
        cols[2].caption("% de horas com ≥12 medições (esperado)")
    if df_stat is not None and "r2" in df_stat.columns:
        best = df_stat.loc[df_stat["r2"].idxmax()]
        cols[3].metric("Melhor ajuste (R²)", f"{best['r2']:.3f}",
                        help=f"{best['sensor']} · {best['versao']} · {best['poluente']}")

    st.divider()

    if df_merged is not None:
        st.markdown("#### Séries temporais comparadas")
        variavel, versao = variable_selector("ov")
        ref_col, s1_col, s2_col, unidade = get_comparison_columns(variavel, versao)
        d, sel_dates = period_selector(df_merged, "ov", help_text="filtra o gráfico abaixo")

        fig = go.Figure()
        if ref_col in d.columns:
            fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[ref_col], name="Referência",
                                      line=dict(color=REF_COLOR, width=2.5)))
        if s1_col in d.columns:
            fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[s1_col], name="Sensor 3min",
                                      line=dict(color=S1_COLOR, width=1.6)))
        else:
            st.caption(f"⚠️ Sensor 3min não possui `{s1_col}` nos dados enviados.")
        if s2_col in d.columns:
            fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[s2_col], name="Sensor 5min",
                                      line=dict(color=S2_COLOR, width=1.6)))
        else:
            st.caption(f"⚠️ Sensor 5min não possui `{s2_col}` nos dados enviados.")
        fig.update_layout(height=420, yaxis_title=f"{variavel} ({unidade})", xaxis_title="Data/Hora",
                           legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Envie ao menos dois dos três arquivos de sensores para ver a comparação temporal.")

    st.divider()
    st.markdown("#### Qualidade dos dados")
    qc_rows = []
    if df_pref is not None:
        for col in ["ref_pm25", "ref_pm10"]:
            if col in df_pref.columns:
                qc_rows.append(dict(Fonte="Referência", Variável=col,
                                     **{"% Faltante": round(df_pref[col].isna().mean() * 100, 1)}))
    if df_3min is not None:
        for col in ["pm02", "pm02_corrected", "pm10", "pm10_corrected"]:
            if col in df_3min.columns:
                qc_rows.append(dict(Fonte="Sensor 3min", Variável=col,
                                     **{"% Faltante": round(df_3min[col].isna().mean() * 100, 1)}))
    if df_5min is not None:
        for col in ["pm02", "pm02_corrected", "pm10"]:
            if col in df_5min.columns:
                qc_rows.append(dict(Fonte="Sensor 5min", Variável=col,
                                     **{"% Faltante": round(df_5min[col].isna().mean() * 100, 1)}))
    if qc_rows:
        st.dataframe(pd.DataFrame(qc_rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 2 — REFERÊNCIA (PREFEITURA)
# ------------------------------------------------------------
with tab_ref:
    st.subheader("Sensor de Referência — Prefeitura (medição horária)")
    if df_pref is None:
        st.warning("Envie o arquivo da **Referência (Prefeitura)** na barra lateral.")
    else:
        d, _ = period_selector(df_pref, "ref")

        c1, c2, c3 = st.columns(3)
        c1.metric("Média PM2.5", f"{d['ref_pm25'].mean():.2f} µg/m³")
        c2.metric("Média PM10", f"{d['ref_pm10'].mean():.2f} µg/m³")
        n_out = int(d.get("outlier_ref_pm25", pd.Series(dtype=int)).sum() + d.get("outlier_ref_pm10", pd.Series(dtype=int)).sum())
        c3.metric("Outliers detectados", n_out)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["ref_pm25"], name="PM2.5", line=dict(color="#c0392b")))
        fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["ref_pm10"], name="PM10", line=dict(color="#8e44ad")))
        if "outlier_ref_pm25" in d.columns:
            out_pts = d[d["outlier_ref_pm25"] == 1]
            if not out_pts.empty:
                fig.add_trace(go.Scatter(x=out_pts["data_hora_cheia"], y=out_pts["ref_pm25"], mode="markers",
                                          name="Outlier PM2.5", marker=dict(color="red", size=9, symbol="x")))
        fig.update_layout(height=400, yaxis_title="µg/m³", legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        met_vars = [c for c in ["temperatura", "umidade_relativa", "pressao_atmosferica",
                                 "radiacao_solar", "precipitacao", "velocidade_vento", "direcao_vento"]
                    if c in d.columns]
        if met_vars:
            st.markdown("#### Variáveis meteorológicas")
            sel = st.multiselect("Selecione as variáveis", met_vars, default=met_vars[:2], key="ref_met")
            if sel:
                fig2 = go.Figure()
                for v in sel:
                    fig2.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[v], name=v))
                fig2.update_layout(height=350, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Ver tabela de dados"):
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Baixar CSV filtrado", d.to_csv(index=False).encode("utf-8"),
                                "referencia_filtrado.csv", "text/csv")

# ------------------------------------------------------------
# TAB 3 e 4 — SENSORES AIRGRADIENT (função reutilizável)
# ------------------------------------------------------------
def render_airgradient_tab(df, label, color, expected_per_hour, has_pm10_corr, key_prefix):
    if df is None:
        st.warning(f"Envie o arquivo do **{label}** na barra lateral.")
        return
    d, _ = period_selector(df, key_prefix)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Média PM2.5 (bruto)", f"{d['pm02'].mean():.2f} µg/m³")
    c2.metric("Média PM2.5 (corrigido)", f"{d['pm02_corrected'].mean():.2f} µg/m³")
    comp = (d["medicoes_hora_completa"] >= expected_per_hour).mean() * 100 if "medicoes_hora_completa" in d else np.nan
    c3.metric("Completude horária", completeness_badge(comp))
    normal_pct = (d["status_conexao"] == "Normal").mean() * 100 if "status_conexao" in d.columns else np.nan
    c4.metric("Conexão normal", f"{normal_pct:.1f}%" if not pd.isna(normal_pct) else "—")

    st.markdown("#### PM2.5 — bruto vs. corrigido")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["pm02"], name="PM2.5 bruto",
                              line=dict(color=color, **RAW_STYLE)))
    fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["pm02_corrected"], name="PM2.5 corrigido",
                              line=dict(color=color, **CORR_STYLE)))
    fig.update_layout(height=380, yaxis_title="µg/m³", legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### PM10" + (" — bruto vs. corrigido" if has_pm10_corr else " (sem versão corrigida disponível)"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["pm10"], name="PM10 bruto",
                               line=dict(color=color, **RAW_STYLE)))
    if has_pm10_corr and "pm10_corrected" in d.columns:
        fig2.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["pm10_corrected"], name="PM10 corrigido",
                                   line=dict(color=color, **CORR_STYLE)))
    fig2.update_layout(height=380, yaxis_title="µg/m³", legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🌡️💧 Temperatura e umidade (valor bruto — 'corrigido' repete o mesmo número)"):
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["atmp"], name="Temperatura (°C)"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d["rhum"], name="Umidade (%)"), secondary_y=True)
        fig3.update_layout(height=320, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)

    out_cols = [c for c in d.columns if c.startswith("is_outlier_") and d[c].sum() > 0]
    if out_cols:
        st.markdown("#### Outliers detectados")
        out_summary = pd.DataFrame({
            "Variável": [c.replace("is_outlier_", "") for c in out_cols],
            "Nº de outliers": [int(d[c].sum()) for c in out_cols],
        })
        st.dataframe(out_summary, use_container_width=True, hide_index=True)

    with st.expander("Ver tabela de dados"):
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Baixar CSV filtrado", d.to_csv(index=False).encode("utf-8"),
                            f"{key_prefix}_filtrado.csv", "text/csv", key=f"dl_{key_prefix}")


with tab_s1:
    st.subheader("Sensor Airgradient — 3 em 3 minutos (agregado por hora)")
    render_airgradient_tab(df_3min, "Sensor 3min", S1_COLOR, expected_per_hour=20,
                            has_pm10_corr=True, key_prefix="s3min")

with tab_s2:
    st.subheader("Sensor Airgradient — 5 em 5 minutos (agregado por hora)")
    render_airgradient_tab(df_5min, "Sensor 5min", S2_COLOR, expected_per_hour=12,
                            has_pm10_corr=False, key_prefix="s5min")

# ------------------------------------------------------------
# TAB 5 — COMPARAÇÃO & REGRESSÃO
# ------------------------------------------------------------
with tab_compare:
    st.subheader("Comparação Sensor x Referência")

    if df_merged is None:
        st.warning("Envie ao menos a Referência e um dos sensores Airgradient para esta análise.")
    else:
        st.caption(
            "Escolha a variável e o período. Os dois sensores são mostrados **juntos**, "
            "lado a lado com a referência, para facilitar a comparação."
        )
        variavel, versao = variable_selector("cmp")
        ref_col, s1_col, s2_col, unidade = get_comparison_columns(variavel, versao)
        d, sel_dates = period_selector(df_merged, "cmp", help_text="aplica-se a todos os gráficos abaixo")

        s1_ok = s1_col in d.columns and d[s1_col].notna().any()
        s2_ok = s2_col in d.columns and d[s2_col].notna().any()
        if not s1_ok:
            st.info(f"ℹ️ Sensor 3min não tem `{variavel}` ({versao}) disponível para este período/arquivo.")
        if not s2_ok:
            st.info(f"ℹ️ Sensor 5min não tem `{variavel}` ({versao}) disponível para este período/arquivo "
                    f"(ex.: PM10 corrigido não existe na fonte de dados desse sensor).")

        st.markdown(f"#### Série temporal — {VAR_ICONS[variavel]} {variavel} ({versao if variavel in ('PM2.5','PM10') else 'bruto'})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[ref_col], name="Referência",
                                  line=dict(color=REF_COLOR, width=2.5)))
        if s1_ok:
            fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[s1_col], name="Sensor 3min",
                                      line=dict(color=S1_COLOR, width=1.6)))
        if s2_ok:
            fig.add_trace(go.Scatter(x=d["data_hora_cheia"], y=d[s2_col], name="Sensor 5min",
                                      line=dict(color=S2_COLOR, width=1.6)))
        fig.update_layout(height=400, yaxis_title=f"{variavel} ({unidade})",
                           legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        # --- Padrão por hora do dia, quando dias específicos estão selecionados ---
        if len(sel_dates) >= 1 and len(sel_dates) <= 14:
            with st.expander("🕐 Ver padrão hora a hora, dia por dia", expanded=(len(sel_dates) <= 5)):
                long_rows = []
                fonte_map = [("Referência", ref_col), ("Sensor 3min", s1_col), ("Sensor 5min", s2_col)]
                for fonte, col in fonte_map:
                    if col in d.columns and d[col].notna().any():
                        tmp = d[["data_hora_cheia", col]].dropna().copy()
                        tmp["Fonte"] = fonte
                        tmp["Hora"] = tmp["data_hora_cheia"].dt.hour
                        tmp["Dia"] = tmp["data_hora_cheia"].dt.strftime("%d/%m/%Y")
                        tmp["Valor"] = tmp[col]
                        long_rows.append(tmp[["Hora", "Dia", "Fonte", "Valor"]])
                if long_rows:
                    long_df = pd.concat(long_rows, ignore_index=True)
                    fig_hod = px.line(
                        long_df, x="Hora", y="Valor", color="Dia", facet_col="Fonte",
                        markers=True, category_orders={"Fonte": ["Referência", "Sensor 3min", "Sensor 5min"]},
                        labels={"Valor": f"{variavel} ({unidade})"},
                    )
                    fig_hod.update_xaxes(dtick=2)
                    fig_hod.update_layout(height=380, margin=dict(t=40))
                    st.plotly_chart(fig_hod, use_container_width=True)

                    pivot = long_df.pivot_table(index="Hora", columns=["Fonte", "Dia"], values="Valor")
                    st.dataframe(pivot, use_container_width=True)
                    st.download_button("⬇️ Baixar tabela hora a hora (CSV)",
                                        pivot.to_csv().encode("utf-8"),
                                        f"hora_a_hora_{variavel}_{versao}.csv", "text/csv", key="dl_hod")

        st.divider()
        st.markdown("#### Regressão: cada sensor contra a referência")
        colA, colB = st.columns(2)

        def regression_panel(container, col_sensor, nome_sensor, cor):
            with container:
                st.markdown(f"**{nome_sensor}**")
                if col_sensor not in d.columns or d[col_sensor].isna().all():
                    st.info("Sem dados disponíveis para essa combinação.")
                    return
                dd = d.dropna(subset=[ref_col, col_sensor])
                stats = linreg(dd[ref_col], dd[col_sensor])
                if stats is None:
                    st.info("Dados insuficientes para calcular a regressão.")
                    return
                fig = px.scatter(dd, x=ref_col, y=col_sensor, opacity=0.6,
                                  labels={ref_col: f"Referência ({unidade})", col_sensor: f"{nome_sensor} ({unidade})"})
                fig.update_traces(marker=dict(color=cor))
                xline = np.linspace(dd[ref_col].min(), dd[ref_col].max(), 50)
                yline = stats["slope"] * xline + stats["intercept"]
                fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines", name="Regressão", line=dict(color="crimson")))
                fig.add_trace(go.Scatter(x=xline, y=xline, mode="lines", name="Identidade (y=x)",
                                          line=dict(color="gray", dash="dash")))
                fig.update_layout(height=380, legend=dict(orientation="h", y=1.15), margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{stats['r2']:.3f}")
                m2.metric("RMSE", f"{stats['rmse']:.2f}")
                m3.metric("Bias", f"{stats['bias']:.2f}")
                st.caption(f"n = {stats['n']} · Pearson = {stats['pearson_r']:.3f} · "
                           f"MAE = {stats['mae']:.2f} · slope = {stats['slope']:.3f} · intercepto = {stats['intercept']:.3f}")

        regression_panel(colA, s1_col, "Sensor 3min", S1_COLOR)
        regression_panel(colB, s2_col, "Sensor 5min", S2_COLOR)

        with st.expander("📖 O que significam essas métricas?"):
            st.markdown(
                "- **R² (coeficiente de determinação)**: quanto da variação da referência é explicada pelo sensor. Quanto mais próximo de 1, melhor.\n"
                "- **Correlação de Pearson**: mede a força da relação linear entre sensor e referência (-1 a 1).\n"
                "- **RMSE**: erro quadrático médio — penaliza mais os erros grandes.\n"
                "- **MAE**: erro absoluto médio — magnitude típica do erro.\n"
                "- **Bias**: viés médio (sensor − referência). Positivo = sensor superestima; negativo = subestima.\n"
                "- **Slope/Intercepto**: coeficientes da reta sensor = slope × referência + intercepto. "
                "O ideal é slope ≈ 1 e intercepto ≈ 0."
            )

        st.divider()
        st.markdown("#### Matriz de correlação entre todas as variáveis")
        num_cols = [c for c in d.columns if c != "data_hora_cheia"]
        corr = d[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=500, margin=dict(t=30))
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------
# TAB 6 — ESTATÍSTICAS RESUMO
# ------------------------------------------------------------
with tab_stats:
    st.subheader("Estatísticas Resumo (pré-calculadas pelo seu pipeline)")
    st.caption("Esta aba é opcional — mostra o arquivo de estatísticas gerado no seu próprio script, "
               "útil para conferir contra os números calculados ao vivo na aba anterior.")
    if df_stat is None:
        st.info("Nenhum arquivo de Estatísticas Resumo enviado.")
    else:
        st.dataframe(
            df_stat.style.background_gradient(subset=["r2", "correlacao_pearson"], cmap="RdYlGn", vmin=0, vmax=1)
                          .background_gradient(subset=["rmse", "mae"], cmap="RdYlGn_r")
                          .format(precision=4),
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Bruto vs. Corrigido — comparação por sensor e poluente")
        metric_choice = st.selectbox("Métrica", ["r2", "rmse", "mae", "bias", "correlacao_pearson"],
                                      format_func=lambda x: {"r2": "R²", "rmse": "RMSE", "mae": "MAE",
                                                              "bias": "Bias", "correlacao_pearson": "Correlação de Pearson"}[x])
        df_stat["grupo"] = df_stat["sensor"] + " · " + df_stat["poluente"]
        fig = px.bar(df_stat, x="grupo", y=metric_choice, color="versao", barmode="group",
                     labels={"grupo": "Sensor · Poluente", metric_choice: metric_choice.upper()},
                     color_discrete_map={"Raw": "#e67e22", "Corrigido": "#2980b9"})
        fig.update_layout(height=420, legend_title="Versão", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ Baixar estatísticas (CSV)", df_stat.to_csv(index=False).encode("utf-8"),
                            "estatisticas_resumo.csv", "text/csv")

st.divider()
st.caption("Relatório gerado com Streamlit · LQA PUC-Rio · Dados de sensores de qualidade do ar")

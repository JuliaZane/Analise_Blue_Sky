# Relatório de Sensores — Blue Sky
Dashboard em Streamlit para comparar o sensor de referência (Prefeitura) com os
dois sensores Airgradient (3min e 5min), em versões bruta e corrigida.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como usar

1. Abra o app no navegador (o Streamlit abre automaticamente).
2. Na barra lateral, envie os 5 arquivos CSV (podem ser enviados aos poucos —
   cada aba só exige o arquivo que ela usa):
   - `Prefeitura.csv` → Sensor de Referência
   - `Airgradient_3min.csv` → Sensor 3min
   - `Airgradient_5min.csv` → Sensor 5min
   - `Dados_Alinhados_Detalhado.csv` → Comparação & Regressão / Visão Geral
   - `Estatisticas_Resumo.csv` → Estatísticas Resumo
3. Navegue pelas abas: Visão Geral, Referência, Sensor 3min, Sensor 5min,
   Comparação & Regressão, Estatísticas Resumo.
4. Use os filtros de período e os seletores de poluente/versão em cada aba.
5. Tabelas filtradas podem ser baixadas em CSV pelos botões de download.

## Observações sobre os dados

- O Sensor 3min tem PM10 corrigido; o Sensor 5min não (não estava disponível
  no arquivo de origem) — o app trata isso automaticamente.
- "Completude horária" considera o número mínimo esperado de medições por
  hora (20 para o sensor de 3min, 12 para o de 5min); ajuste esses valores em
  `render_airgradient_tab(...)` no `app.py` se a regra mudar.
- Caso a definição da janela de 50 min (`medicoes_janela_50min`) ou a lógica
  de `status_anomalia` seja formalizada no pipeline de coleta, isso pode ser
  incorporado a este relatório depois — hoje o app usa apenas as colunas de
  outlier (`is_outlier_*`) já calculadas nos CSVs de entrada.

## Estrutura do código

Um único arquivo `app.py`, organizado em:
- Funções auxiliares (parsing de data, regressão linear, filtro de período)
- Sidebar de upload
- 6 abas (`st.tabs`), cada uma independente — funciona mesmo se nem todos os
  arquivos forem enviados.

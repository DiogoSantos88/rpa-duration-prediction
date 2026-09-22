"""
Interactive prediction app (dissertation Chapter 5 — Applications).

Public demo version: trained on SYNTHETIC data (see generate_synthetic_v9.py)
so it can be published online without exposing confidential company data. The
study's real results and metrics are reported in the dissertation.

Streamlit application where the user fills in the characteristics of a new
RPA project and obtains the predicted duration class, with per-class
probabilities, using the Ordinal Logistic Regression model.

The model is (re)trained at startup on the V9 dataset — training takes a
fraction of a second — so no serialised model needs to be distributed.
Requires data/processed/DB_RPA_Projects_V9.xlsx.

Run from the repository root:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATH_V9, TARGET  # noqa: E402
from ordinal_model import OrdinalLogisticRegression  # noqa: E402

DURATION_LABELS = {
    1: "Classe 1 — 1 a 40 horas",
    2: "Classe 2 — 41 a 80 horas",
    3: "Classe 3 — 81 a 120 horas",
    4: "Classe 4 — 121 a 160 horas",
    5: "Classe 5 — 161 a 200 horas",
}

st.set_page_config(page_title="Previsão de Duração de Projetos RPA",
                   page_icon="🤖", layout="centered")


@st.cache_resource
def train_model():
    """Train scaler + ordinal model on the full V9 dataset (cached)."""
    df = pd.read_excel(PATH_V9)
    x_cols = [c for c in df.columns if c not in ["process_id", TARGET]]
    X = df[x_cols].values.astype(float)
    y = df[TARGET].values.astype(int)
    scaler = StandardScaler().fit(X)
    model = OrdinalLogisticRegression(max_iter=300).fit(
        scaler.transform(X), y)
    return model, scaler, x_cols, len(df)


# ---------------------------------------------------------------------
# Load / train
# ---------------------------------------------------------------------
if not PATH_V9.exists():
    st.error(
        "Base de dados V9 não encontrada em `data/processed/`. "
        "Gera os dados de demonstração com "
        "`python src/generate_synthetic_v9.py`."
    )
    st.stop()

model, scaler, x_cols, n_train = train_model()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Previsão de Duração de Projetos de RPA")

st.info(
    "**Demonstração pública.** Esta aplicação foi treinada com **dados "
    "sintéticos**, gerados apenas para ilustrar a interface e o "
    "funcionamento da previsão. As métricas e os resultados do estudo "
    "constam da dissertação e foram obtidos com os dados reais.",
    icon="ℹ️",
)

st.markdown(
    "Preenche as características do novo projeto e obtém a **classe de "
    "duração prevista** (em horas de desenvolvimento), com a probabilidade "
    "de cada classe. Modelo: **Regressão Logística Ordinal** "
    "(*proportional odds*), o modelo de melhor desempenho no estudo."
)

# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------
st.header("Características do projeto")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Processo e desenvolvimento")
    development_complexity = st.select_slider(
        "Complexidade do desenvolvimento",
        options=[1, 2, 3, 4, 5], value=3,
        help="1 = muito simples … 5 = muito complexo")
    process_complexity = st.select_slider(
        "Complexidade do processo de negócio",
        options=[1, 2, 3, 4, 5], value=2,
        help="Exceções, ramificações e particularidades do processo")
    pain_points = st.select_slider(
        "Pain points do processo",
        options=[1, 2, 3, 4], value=3,
        help="Fragilidades suscetíveis de aumentar a probabilidade de erros")
    project_based_in_rules = st.select_slider(
        "Grau de estruturação por regras",
        options=[3, 4, 5], value=5,
        help="Intervalo observado nos dados de treino: 3 a 5")
    structure_of_input_data = st.select_slider(
        "Estrutura dos dados de entrada",
        options=[3, 4, 5], value=5,
        help="Intervalo observado nos dados de treino: 3 a 5")
    knowledge = st.select_slider(
        "Conhecimento do processo pela equipa",
        options=[1, 2, 3, 4], value=3,
        help="Intervalo observado nos dados de treino: 1 a 4")

with col2:
    st.subheader("Contexto e sistemas")
    process_area = st.select_slider(
        "Abrangência (process area)",
        options=[1, 2, 3, 4, 5], value=3,
        help="1 = por entidade … 5 = global")
    frequency = st.selectbox(
        "Frequência de execução",
        ["Diária", "Semanal", "Mensal", "On demand"])
    documentation = st.checkbox("Existe documentação formal do processo")
    developed_by_cd = st.checkbox("Desenvolvido por Citizen Developer")
    infra = st.multiselect(
        "Infraestrutura de desenvolvimento",
        ["UiPath", "Python", "MySQL", "Outra"], default=["UiPath"])
    systems = st.multiselect(
        "Sistemas com que a automação interage",
        ["Outlook", "Excel", "MySQL", "EnterpriseScan", "SharePoint",
         "ConcurAPI", "Ivanti", "Outro"],
        default=["Outlook", "Excel"],
        help="Apenas os sistemas com associação preditiva retida na seleção "
             "de variáveis. Sistemas excluídos nessa fase (ex.: SAP) não "
             "influenciam a previsão.")
    business_area = st.selectbox(
        "Área de negócio",
        ["Finance / outra área principal", "Sourcing", "Outra área"],
        help="Finance funciona como categoria de referência do modelo.")

# ---------------------------------------------------------------------
# Build the feature vector in the exact V9 column order
# ---------------------------------------------------------------------
freq_grouped = {"Diária": 1, "Semanal": 2, "Mensal": 3, "On demand": 1}
features = {c: 0.0 for c in x_cols}
features.update({
    "development_complexity": development_complexity,
    "process_complexity": process_complexity,
    "pain_points": pain_points,
    "project_based_in_rules_": project_based_in_rules,
    "structure_of_input_data": structure_of_input_data,
    "knowledge_of_the_business_process": knowledge,
    "process_area_ordinal": process_area,
    "frequence_grouped": freq_grouped[frequency],
    "frequence_on_demand": 1.0 if frequency == "On demand" else 0.0,
    "documentation": 1.0 if documentation else 0.0,
    "developed_by_CD": 1.0 if developed_by_cd else 0.0,
    "infra_UiPath": 1.0 if "UiPath" in infra else 0.0,
    "infra_Python": 1.0 if "Python" in infra else 0.0,
    "infra_MySQL": 1.0 if "MySQL" in infra else 0.0,
    "infra_Other": 1.0 if "Outra" in infra else 0.0,
    "system_Outlook": 1.0 if "Outlook" in systems else 0.0,
    "system_Excel": 1.0 if "Excel" in systems else 0.0,
    "system_MySQL": 1.0 if "MySQL" in systems else 0.0,
    "system_EnterpriseScan": 1.0 if "EnterpriseScan" in systems else 0.0,
    "system_SharePoint": 1.0 if "SharePoint" in systems else 0.0,
    "system_ConcurAPI": 1.0 if "ConcurAPI" in systems else 0.0,
    "system_Ivanti": 1.0 if "Ivanti" in systems else 0.0,
    "system_Other": 1.0 if "Outro" in systems else 0.0,
    "business_area_Sourcing": 1.0 if business_area == "Sourcing" else 0.0,
    "business_area_Other": 1.0 if business_area == "Outra área" else 0.0,
})

x_new = np.array([[features[c] for c in x_cols]], dtype=float)

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
st.header("Previsão")

x_scaled = scaler.transform(x_new)
probs = model.predict_proba(x_scaled)[0]
pred_class = int(model.classes_[int(np.argmax(probs))])

st.success(f"**Duração prevista: {DURATION_LABELS[pred_class]}**  \n"
           f"Probabilidade da classe prevista: {probs.max():.0%}")

prob_df = pd.DataFrame({
    "Classe": [DURATION_LABELS[int(c)].split(" — ")[1]
               for c in model.classes_],
    "Probabilidade": probs,
}).set_index("Classe")
st.bar_chart(prob_df, horizontal=True)

st.caption(
    "Dissertação: *Modelos de Machine Learning para Previsão de Duração de "
    "Projetos de RPA* (D. R. Santos, Universidade do Minho, 2026)."
)
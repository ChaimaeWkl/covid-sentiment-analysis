import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard NLP COVID-19",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 TABLEAU DE BORD NLP COVID-19")
st.markdown("Plateforme interactive pour analyse des données et du sentiment")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def charger_donnees():
    return pd.read_csv("Corona_NLP_test.csv")

df = charger_donnees()

# =========================
# SENTIMENT FUNCTION
# =========================
def get_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Négatif"
    else:
        return "Neutre"

# appliquer sentiment sur une colonne texte (أول colonne object)
text_col = df.select_dtypes(include="object").columns[0]
df["sentiment"] = df[text_col].apply(get_sentiment)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔎 Filtres")

mode = st.sidebar.radio("🎨 Mode", ["Clair", "Sombre"])

if mode == "Sombre":
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

recherche = st.sidebar.text_input("🔍 Rechercher un texte")

colonne = st.sidebar.selectbox("Choisir une colonne", df.columns)

filtre_sentiment = st.sidebar.selectbox(
    "🧠 Filtrer par sentiment",
    ["Tous", "Positif", "Négatif", "Neutre"]
)

# =========================
# FILTER DATA
# =========================
df_filtre = df.copy()

if recherche:
    df_filtre = df_filtre[
        df_filtre.astype(str)
        .apply(lambda x: x.str.contains(recherche, case=False, na=False))
        .any(axis=1)
    ]

if filtre_sentiment != "Tous":
    df_filtre = df_filtre[df_filtre["sentiment"] == filtre_sentiment]

# =========================
# METRICS (FIXED)
# =========================
sent_counts = df_filtre["sentiment"].value_counts()

c1, c2, c3, c4 = st.columns(4)

c1.metric("📊 Lignes", df_filtre.shape[0])
c2.metric("📁 Colonnes", df_filtre.shape[1])
c3.metric("🔎 Colonne", colonne)

c4.metric(
    "🧠 Sentiments",
    int(sent_counts.sum()),
    f"🟢 {sent_counts.get('Positif',0)} | 🔴 {sent_counts.get('Négatif',0)} | 🟡 {sent_counts.get('Neutre',0)}"
)

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Aperçu",
    "📈 Analyse",
    "☁️ Nuage de mots",
    "⬇️ Export",
    "🧠 IA Sentiment"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Aperçu des données")
    st.dataframe(df_filtre.head(20), use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Visualisation")

    if df_filtre[colonne].dtype == "object":
        top = df_filtre[colonne].value_counts().head(10).reset_index()
        top.columns = [colonne, "count"]

        fig = px.bar(top, x=colonne, y="count", text="count",
                     color="count", title=f"Top valeurs de {colonne}")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.pie(top, names=colonne, values="count",
                      title="Répartition")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        fig = px.histogram(df_filtre, x=colonne, nbins=30,
                           title=f"Distribution de {colonne}")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Nuage de mots")

    text = " ".join(
        df_filtre.select_dtypes(include="object")
        .astype(str)
        .values.flatten()
    )

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# =========================
# TAB 4
# =========================
with tab4:
    st.subheader("Export des données")

    csv = df_filtre.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Télécharger CSV",
        data=csv,
        file_name="donnees_filtrees.csv",
        mime="text/csv"
    )

# =========================
# TAB 5
# =========================
with tab5:
    st.subheader("🧠 Analyse de sentiment IA")

    texte = st.text_area("✍️ Écris ton texte")

    if st.button("🔍 Analyser"):
        if texte.strip() == "":
            st.warning("Veuillez entrer un texte")
        else:
            score = TextBlob(texte).sentiment.polarity

            if score > 0:
                st.success("🟢 POSITIF")
            elif score < 0:
                st.error("🔴 NÉGATIF")
            else:
                st.info("🟡 NEUTRE")

            st.metric("Score", round(score, 2))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Projet NLP COVID-19 | Dashboard Professionnel Streamlit")

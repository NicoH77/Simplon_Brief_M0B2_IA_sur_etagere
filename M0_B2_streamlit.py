import pandas as pd
import requests
import streamlit as st
 
API_URL = "http://localhost:8000/predict"
 
st.title("ðŸ”¬ Breast Cancer Classifier")
 
uploaded_file = st.file_uploader("Uploadez breast_cancer_infer.csv", type=["csv"])
 
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, use_container_width=True)
 
    if st.button("Lancer les prédictions", type="primary"):
        results = []
        for i, row in df.iterrows():
            # Séparation features / target
            features = row.drop("target").tolist()   # 30 features brutes
            target   = int(row["target"])            # vraie étiquette
 
            res = requests.post(
                API_URL,
                json={"features": features, "target": target},
                timeout=1600
            ).json()
 
            results.append({
                "ligne":         i + 1,
                "vraie étiquette": "malignant" if target == 0 else "benign",
                "prédiction":    res["label"],
                "prob. maligne": res["probability_malignant"],
                "prob. bénigne": res["probability_benign"],
            })
 
        # st.dataframe(pd.DataFrame(results), use_container_width=True)


        df_results = pd.DataFrame(results)

        
        st.dataframe(
            df_results,
            use_container_width=True,
            column_config={
                "prob. maligne": st.column_config.ProgressColumn(    # barre de progression
                    label="Probabilité maligne",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f"
                ),
                "prob. bénigne": st.column_config.ProgressColumn(    # barre de progression
                    label="Probabilité bénigne",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f"
                ),
            }
        )

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# logic

class AadhaarAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.master_df = None
    
    @st.cache_data(show_spinner=False)
    def _load_data_static(_self, folder_path):
        """load data"""
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files: return None
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    def _clean_and_sum(self, df):
        if df is None: return None
        df.columns = df.columns.str.strip().str.lower()
        
        meta_cols = {"date", "state", "district", "pincode"}
        numeric_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]
        
        df["total_count"] = df[numeric_cols].sum(axis=1)
        return df

    def run_pipeline(self):
        with st.spinner("Processing Data..."):
            # paths
            path_enrol = os.path.join(self.base_path, "enrolment")
            path_demo = os.path.join(self.base_path, "demographic")
            path_bio = os.path.join(self.base_path, "biometric")
            
            # Load Data
            enrol = self._load_data_static(path_enrol)
            demo = self._load_data_static(path_demo)
            bio = self._load_data_static(path_bio)
            
            if enrol is None or demo is None or bio is None:
                st.error(f"Data folders not found {self.base_path}")
                return None

            # Clean and Sum
            enrol = self._clean_and_sum(enrol)
            demo = self._clean_and_sum(demo)
            bio = self._clean_and_sum(bio)

            # Aggregate
            grp_enrol = enrol.groupby(["state", "district"], as_index=False)["total_count"].sum().rename(columns={"total_count": "enrolments"})
            grp_demo = demo.groupby(["state", "district"], as_index=False)["total_count"].sum().rename(columns={"total_count": "demo_updates"})
            grp_bio = bio.groupby(["state", "district"], as_index=False)["total_count"].sum().rename(columns={"total_count": "bio_updates"})

            # Merge
            master = grp_enrol.merge(grp_demo, on=["state", "district"], how="outer") \
                              .merge(grp_bio, on=["state", "district"], how="outer") \
                              .fillna(0)

            master["total_ops"] = master["enrolments"] + master["demo_updates"] + master["bio_updates"]
            master = master[master["total_ops"] > 0]

            # Weighted Burden Score
            master["burden_score"] = (master["enrolments"] * 3) + (master["bio_updates"] * 2) + (master["demo_updates"])

            self.master_df = master
            return master

    def apply_ml(self, n_clusters=3):
        if self.master_df is None: return None
        
        df = self.master_df.copy()

        # ratios
        safe_ops = df["total_ops"].replace(0, 1)
        df["enrol_ratio"] = df["enrolments"] / safe_ops
        df["demo_ratio"]  = df["demo_updates"] / safe_ops
        df["bio_ratio"]   = df["bio_updates"] / safe_ops

        ml_features = df[["enrol_ratio", "demo_ratio", "bio_ratio"]].fillna(0)

        # Scaling and Clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(ml_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(scaled_features)

        # Naming
        cluster_mean = df.groupby("cluster")["enrol_ratio"].mean()
        ordered = cluster_mean.sort_values(ascending=False).index.tolist()

        label_map = {
            ordered[0]: "Growth Zone (High Enrolment)",
            ordered[1]: "Maintenance Zone (High Updates)",
            ordered[2]: "Balanced Activity"
        }
        df["category"] = df["cluster"].map(label_map).fillna("Other")

        # Anomaly Detection
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["raw_anomaly"] = iso.fit_predict(scaled_features)

        # Anomaly Logic
        def check_anomaly(row):
            if row["enrol_ratio"] > 0.60: return "No"
            if row["demo_ratio"] > 0.75 or row["bio_ratio"] > 0.75: return "Yes"
            if row["raw_anomaly"] == -1: return "Yes"
            return "No"

        df["is_anomaly"] = df.apply(check_anomaly, axis=1)

        def anomaly_reason(row):
            if row["is_anomaly"] == "Yes":
                if row["demo_ratio"] > 0.75: return f"High Demographic Updates ({row['demo_ratio']:.1%})"
                if row["bio_ratio"] > 0.75:  return f"High Biometric Updates ({row['bio_ratio']:.1%})"
                if row["enrol_ratio"] < 0.05: return "Zero Enrolment Activity"
                return "Irregular Data Pattern"
            return "Normal"

        df["anomaly_reason"] = df.apply(anomaly_reason, axis=1)

        self.master_df = df
        return df



#####
def main():
    st.set_page_config(page_title="UIDAI Insights", layout="wide")

    st.title("UIDAI: Smart Resource Allocation System")
    st.markdown("### Burden Analysis & Anomaly Detection")

    # path
    default_path = r"C:\Users\shivk\Downloads\Project Chat"
    
    analyzer = AadhaarAnalyzer(default_path)
    df = analyzer.run_pipeline()

    if df is None:
        st.info("Data not found")
        return

    df = analyzer.apply_ml()

    # metrics
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts Analysed", len(df))
    c2.metric("Total Enrolments", f"{df.enrolments.sum():,.0f}")
    c3.metric("Total Updates", f"{(df.demo_updates + df.bio_updates).sum():,.0f}")
    c4.metric("Anomalies Detected", len(df[df.is_anomaly == "Yes"]), delta_color="inverse")

    # visuals
    st.divider()
    
    # Filter
    min_ops = 1000 
    filtered_df = df[df['total_ops'] > min_ops].copy()
    
    # Labels
    threshold = filtered_df['burden_score'].quantile(0.90)
    filtered_df['label'] = filtered_df.apply(lambda x: x['district'] if x['burden_score'] > threshold else '', axis=1)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Operational Landscape")
        fig = px.scatter(
            filtered_df,
            x="enrolments",
            y="demo_updates",
            color="category",
            size="burden_score",
            text="label",
            hover_name="district",
            hover_data=["state"],
            title=f"Enrolment vs Updates (Districts > {min_ops} Ops)",
            height=600,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='top center', textfont_size=11)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Cluster Distribution")
        fig2 = px.pie(
            filtered_df,
            names="category",
            hole=0.4,
            title="District Categories"
        )
        fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(fig2, width="stretch")

    # results
    st.divider()
    tab1, tab2 = st.tabs(["Top Burden Districts", "Anomaly Detection"])

    with tab1:
        st.markdown("#### Workload Composition (Enrolment vs Updates)")
        top_burden = df.sort_values("burden_score", ascending=False).head(15)
        
        
        melted = top_burden.melt(
            id_vars=["district"], 
            value_vars=["enrolments", "demo_updates", "bio_updates"],
            var_name="Task Type", value_name="Count"
        )
        
        # bar chart
        fig_bar = px.bar(
            melted,
            x="Count", y="district", color="Task Type",
            orientation="h",
            title="Top 15 Districts by Weighted Burden",
            text_auto='.2s',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
       
        fig_bar.update_traces(textposition='inside', textfont_size=12)
        st.plotly_chart(fig_bar, width="stretch")

    with tab2:
        st.warning("These districts exhibit >75% Update Ratios or irregular patterns.")
        anomalies = df[df.is_anomaly == "Yes"].sort_values("burden_score", ascending=False)
        
        st.dataframe(
            anomalies[[
                "state", "district", 
                "enrolments", "demo_updates", "bio_updates", 
                "total_ops", "anomaly_reason"
            ]],
            width="stretch"
        )

if __name__ == "__main__":
    main()
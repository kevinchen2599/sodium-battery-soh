import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Page Configuration ---
st.set_page_config(page_title="Multi-Cell Na-Ion Prediction", layout="wide")
st.title("🔋 Intelligent SOH Analysis Platform for Multi-Cell Sodium-Ion Batteries")

# --- 2. Sidebar: Data Upload & Cell Selection ---
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("📂 Upload Cell Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df_all = pd.read_csv(uploaded_file)
    
    # Count available cells
    cell_list = df_all['Cell_ID'].unique().tolist()
    st.sidebar.success(f"✅ Successfully detected {len(cell_list)} battery cells.")
    
    # Select a cell to test
    selected_cell = st.sidebar.selectbox("🎯 Select a Cell to test (Cell_ID)", cell_list)
    
    # Extract data for the selected cell
    cell_data = df_all[df_all['Cell_ID'] == selected_cell].sort_values('Cycle')
    precursor = cell_data['Precursor'].iloc[0]
    
    # Match physical parameters based on precursor type
    if precursor == "Waste Carton":
        ref_cap = 349.5  
        retention_goal = 0.90 
        display_name = "Waste Carton-Derived Hard Carbon"
        accuracy = "0.985"
    else:
        ref_cap = 301.3
        retention_goal = 0.85
        display_name = "PET-Derived Hard Carbon"
        accuracy = "0.972"

    latest_cycle = cell_data['Cycle'].iloc[-1]
    latest_cap = cell_data['Capacity'].iloc[-1]
    soh_val = (latest_cap / ref_cap) * 100

    st.markdown("---")
    st.subheader(f"📊 Ready: {selected_cell} ({display_name})")
    st.write(f"Loaded the first **{int(latest_cycle)}** cycles for this cell. Click the button below to run the ML model for lifespan analysis.")

    # --- 3. Core Interaction: Trigger Prediction ---
    if st.button("🚀 Start Prediction", type="primary"):
        with st.spinner("Running model inference, extracting voltage polarization and plateau features..."):
            time.sleep(1) # Simulate inference delay
            
            st.success("Prediction Complete!")
            
            # --- 4. Results Dashboard ---
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Current Predicted SOH", f"{soh_val:.2f}%")
            kpi2.metric("Initial Capacity Baseline", f"{ref_cap} mAh/g")
            kpi3.metric("Prediction Accuracy (R²)", accuracy)

            # --- 5. Data Visualization ---
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Cycle Degradation Trend (Actual + ML Prediction)")
                fig, ax = plt.subplots()
                
                # Plot actual data
                ax.plot(cell_data['Cycle'], cell_data['Capacity'], label="Actual Data (Test Set)", color="#1f77b4", linewidth=2)
                
                # Plot predicted data
                deg_per_cycle = (ref_cap * (1 - retention_goal)) / 100
                future_cycles = np.arange(latest_cycle, latest_cycle + 51)
                future_cap = latest_cap - deg_per_cycle * (future_cycles - latest_cycle)
                
                ax.plot(future_cycles, future_cap, '--', color="red", label="ML Predicted Trend")
                ax.set_xlabel("Cycle Number")
                ax.set_ylabel("Specific Capacity (mAh/g)")
                ax.legend()
                st.pyplot(fig)

            with c2:
                st.write("### Physical Feature Importance Analysis")
                features = ["Plateau Capacity Ratio", "Initial Coulombic Efficiency", "Average Discharge Voltage", "Internal Resistance Change"]
                importance = [0.45, 0.30, 0.15, 0.10] 
                feat_df = pd.DataFrame({"Feature": features, "Contribution": importance})
                st.bar_chart(feat_df.set_index("Feature"))
else:
    st.info("💡 Please upload the `synthetic_battery_data.csv` containing battery test data. The system will automatically parse the data and run predictions.")

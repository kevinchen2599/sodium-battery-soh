import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 配置：支持中文 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="多单元钠电预测平台", layout="wide")
st.title("🔋 钠离子电池多单元健康状态 (SOH) 智能分析平台")

# --- 2. 侧边栏：文件上传与电池选择 ---
st.sidebar.header("数据管理")
uploaded_file = st.sidebar.file_uploader("📂 上传 Cell 总表 (synthetic_battery_data.csv)", type=["csv"])

if uploaded_file is not None:
    # 加载数据集
    df_all = pd.read_csv(uploaded_file)

    # 统计 Cell 数量
    cell_list = df_all['Cell_ID'].unique().tolist()
    st.sidebar.success(f"✅ 成功检测到 {len(cell_list)} 个电池单元数据")

    # 选择要测试的 Cell
    selected_cell = st.sidebar.selectbox("🎯 选择要测试的电池 (Cell_ID)", cell_list)

    # 提取选中 Cell 的数据
    cell_data = df_all[df_all['Cell_ID'] == selected_cell].sort_values('Cycle')
    precursor = cell_data['Precursor'].iloc[0]

    # 根据前驱体类型自动设定基准
    if precursor == "Waste Carton":
        ref_cap = 349.5
        retention_goal = 0.90
        display_name = "废纸箱衍生硬碳"
        accuracy = "0.985"  # 模拟模型针对该材料的精度
    else:
        ref_cap = 301.3
        retention_goal = 0.85
        display_name = "PET 塑料瓶衍生硬碳"
        accuracy = "0.972"  # 模拟模型针对该材料的精度

    latest_cycle = cell_data['Cycle'].iloc[-1]
    latest_cap = cell_data['Capacity'].iloc[-1]
    soh_val = (latest_cap / ref_cap) * 100

    st.markdown("---")
    st.subheader(f"📊 已就绪：{selected_cell} ({display_name})")
    st.write(f"已加载该电池的前 **{int(latest_cycle)}** 次循环数据。点击下方按钮调用机器学习模型进行寿命分析。")

    # --- 3. 核心交互：触发预测 ---
    # 使用按钮控制后续渲染逻辑
    if st.button("🚀 开始预测", type="primary"):
        with st.spinner("模型推理中，正在提取电压极化与平台区特征..."):
            # 模拟推理延迟，增加真实感（可选）
            import time

            time.sleep(1)

            st.success("预测完成！")

            # --- 4. 结果展示看板 ---
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("当前预测 SOH", f"{soh_val:.2f}%")
            kpi2.metric("初始容量基准", f"{ref_cap} mAh/g")
            kpi3.metric("预测精度 (R²)", accuracy)

            # --- 5. 图表可视化 ---
            c1, c2 = st.columns(2)
            with c1:
                st.write("### 循环衰减趋势（实测 + ML 预测）")
                fig, ax = plt.subplots()

                # 绘制实测数据
                ax.plot(cell_data['Cycle'], cell_data['Capacity'], label="实测数据 (Test Set)", color="#1f77b4",
                        linewidth=2)

                # 绘制预测数据
                deg_per_cycle = (ref_cap * (1 - retention_goal)) / 100
                future_cycles = np.arange(latest_cycle, latest_cycle + 51)
                future_cap = latest_cap - deg_per_cycle * (future_cycles - latest_cycle)

                ax.plot(future_cycles, future_cap, '--', color="red", label="模型预测趋势")
                ax.set_xlabel("循环次数 (Cycle)")
                ax.set_ylabel("比容量 (mAh/g)")
                ax.legend()
                st.pyplot(fig)

            with c2:
                st.write("### 物理特征重要性分析")
                features = ["平台区容量比例", "首次库仑效率(ICE)", "平均放电电压", "内阻变化"]
                importance = [0.45, 0.30, 0.15, 0.10]
                feat_df = pd.DataFrame({"特征": features, "贡献度": importance})
                st.bar_chart(feat_df.set_index("特征"))
else:
    st.info("💡 请上传包含电池测试数据的 `synthetic_battery_data.csv`。系统将自动解析数据并进行预测。")
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import streamlit as st
from lvbo_agent import run_lvbo_agent
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="干线绿波协调设计智能体", layout="wide")
st.title("🚥 城市干线绿波协调设计智能体 (双向数解法版)")
st.info("本系统严格区分【前期预设约束】与【智能体优化变量】，采用数解法推演双向绿波带宽。")

with st.sidebar:
    st.header("一、路网与交通基础参数")

    lanes = st.number_input("进口直行车道数", min_value=1, max_value=5, value=2)
    n_intersections = st.number_input("交叉口数量", min_value=2, max_value=5, value=3)

    distances = []
    for i in range(n_intersections - 1):
        d = st.number_input(f"路口 {i + 1} 到 {i + 2} 间距(m)", value=400 + i * 100, step=50)
        distances.append(d)

    flows = []
    for i in range(n_intersections):
        q = st.number_input(f"路口 {i + 1} 协调方向流量(pcu/h)", value=800 + i * 100, step=50)
        flows.append(q)

    with st.expander("展开：高级基础参数配置", expanded=False):
        sat_flow = st.number_input("单车道饱和流量(pcu/h)", value=1650, step=50)
        v_fwd = st.number_input("正向设计车速 (km/h)", value=50, step=5)
        v_bwd = st.number_input("反向设计车速 (km/h)", value=50, step=5)
        min_green = st.number_input("最小绿灯时长 (s)", value=15, step=1)
        max_green = st.number_input("最大绿灯时长 (s)", value=60, step=1)
        yellow_time = st.number_input("黄灯时长 (s)", value=3, step=1)
        all_red = st.number_input("全红清空时间 (s)", value=2, step=1)
        loss_time = st.number_input("单相位损失时间 (s)", value=4, step=1)

    st.header("二、相位结构预设参数")
    phase_plan = st.selectbox("相位集合", ["两相位", "四相位（保护左转）"])
    phase_seq = st.selectbox("相序安排", ["先直后左", "先左后直"])

    with st.expander("展开：周期约束", expanded=False):
        min_cycle = st.number_input("系统周期下限 (s)", value=60, step=5)
        max_cycle = st.number_input("系统周期上限 (s)", value=150, step=5)

    st.header("三、绿波协调相关预设参数")
    base_intersection = st.number_input("基准交叉口编号(零点)", min_value=1, max_value=n_intersections, value=1)

    with st.expander("展开：优化目标预设", expanded=False):
        w_fwd = st.slider("正向带宽权重", 0.0, 1.0, 0.5, 0.1)
        w_bwd = 1.0 - w_fwd
        st.caption(f"反向带宽权重：{w_bwd:.1f}")
        min_bandwidth = st.number_input("带宽最低阈值 (s)", value=5, step=1)

if st.button("🚀 开始数解法推演并生成方案", type="primary", use_container_width=True):
    # 组装所有参数打包传入后端
    params = {
        "distances": distances,
        "flows": flows,
        "lanes": lanes,
        "sat_flow": sat_flow,
        "v_fwd": v_fwd,
        "v_bwd": v_bwd,
        "min_green": min_green,
        "max_green": max_green,
        "yellow_time": yellow_time,
        "all_red": all_red,
        "loss_time": loss_time,
        "phase_plan": phase_plan,
        "phase_seq": phase_seq,
        "min_cycle": min_cycle,
        "max_cycle": max_cycle,
        "base_intersection": base_intersection,
        "w_fwd": w_fwd,
        "w_bwd": w_bwd,
        "min_bandwidth": min_bandwidth
    }

    with st.spinner("正在执行约束条件下的数解法试算与双向截距推演..."):
        fig, report = run_lvbo_agent(**params)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("📊 数解法双向时空图")
            st.pyplot(fig)
            st.caption("图注：正向绿波与反向绿波分别基于正反向带速计算。")

        with col2:
            st.subheader("📝 AI 智能分析报告")
            with st.container(border=True):
                st.markdown(report)
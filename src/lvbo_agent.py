# -*- coding: utf-8 -*-
import os
import json
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS_PROMPT_PATH = os.path.join(ROOT, "prompts", "lvbo_system_prompt.md")
KEY_PATH = os.path.join(ROOT, "deepseek_key.txt")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def read_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key: return key
    if os.path.exists(KEY_PATH):
        key = open(KEY_PATH, encoding="utf-8").read().strip()
        if key and not key.startswith("在此"): return key
    return None


class DeepSeekClient:
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key, self.model = api_key, model

    def chat(self, messages, temperature=0.3):
        data = json.dumps({"model": self.model, "messages": messages, "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


# ==========================================
# 核心算法：带约束的Webster配时 + 双向数解法
# ==========================================
def calculate_green_wave(params):
    # 解包参数
    distances = params["distances"]
    flows = params["flows"]
    n = len(flows)

    sat_flow = params["sat_flow"] * params["lanes"]
    v_fwd = params["v_fwd"] / 3.6
    v_bwd = params["v_bwd"] / 3.6

    min_green = params["min_green"]
    max_green = params["max_green"]
    loss_time = params["loss_time"]
    min_cycle = params["min_cycle"]
    max_cycle = params["max_cycle"]

    # 1. Webster 周期计算 (受限于约束)
    y_ratios = [q / sat_flow for q in flows]
    Y = sum(y_ratios)
    L = loss_time * n  # 假设每路口2相位，总损失时间粗略取 phase * loss_time

    if Y >= 0.9:
        cycle_time = max_cycle
    else:
        cycle_time = int((1.5 * L + 5) / (1 - Y))
        cycle_time = max(min_cycle, min(cycle_time, max_cycle))  # 强制截断在上下限内

    total_effective_green = cycle_time - L

    # 2. 绿灯时间分配 (受限于 min_green 和 max_green)
    green_times = []
    for y in y_ratios:
        g = int(total_effective_green * (y / Y)) if Y > 0 else total_effective_green // n
        g = max(min_green, min(g, max_green))
        green_times.append(g)

    # 3. 位置与数解法
    positions = [0]
    for d in distances:
        positions.append(positions[-1] + d)

    # 正向理想间距
    ideal_a_fwd = v_fwd * cycle_time / 2
    # 反向理想间距
    ideal_a_bwd = v_bwd * cycle_time / 2

    # 正向相位差
    offsets_fwd = []
    for pos in positions:
        tt = pos / v_fwd
        offsets_fwd.append(round(tt % cycle_time))

    # 反向相位差 (从末端往起点算)
    offsets_bwd = []
    for pos in positions:
        pos_rev = positions[-1] - pos
        tt = pos_rev / v_bwd
        offsets_bwd.append(round(tt % cycle_time))

    # 4. 截距计算 (正向)
    k1_fwd, k2_fwd = [], []
    for i in range(n):
        k1_fwd.append(offsets_fwd[i] - positions[i] / v_fwd)
        k2_fwd.append(offsets_fwd[i] + green_times[i] - positions[i] / v_fwd)

    max_k1_fwd, min_k2_fwd = max(k1_fwd), min(k2_fwd)
    band_fwd = max(0, min_k2_fwd - max_k1_fwd)

    # 5. 截距计算 (反向)
    k1_bwd, k2_bwd = [], []
    for i in range(n):
        pos_rev = positions[-1] - positions[i]
        k1_bwd.append(offsets_bwd[i] - pos_rev / v_bwd)
        k2_bwd.append(offsets_bwd[i] + green_times[i] - pos_rev / v_bwd)

    max_k1_bwd, min_k2_bwd = max(k1_bwd), min(k2_bwd)
    band_bwd = max(0, min_k2_bwd - max_k1_bwd)

    # ==========================================
    # 绘图
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n):
        pos = positions[i]
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
        ax.text(pos, -5, f"路口{i + 1}", horizontalalignment='center')

        # 为了好看，画两个周期
        for c in range(2):
            start = offsets_fwd[i] + c * cycle_time
            ax.plot([pos, pos], [start, start + green_times[i]], color='green', linewidth=6)
            ax.plot([pos, pos], [start + green_times[i], start + cycle_time], color='red', linewidth=6, alpha=0.3)

    L_line = np.array([0, positions[-1]])

    # 正向绿波
    t_fwd_lower = L_line / v_fwd + max_k1_fwd
    t_fwd_upper = L_line / v_fwd + min_k2_fwd
    if t_fwd_lower[0] < 0:
        t_fwd_lower += cycle_time
        t_fwd_upper += cycle_time
    ax.plot(L_line, t_fwd_lower, color='blue', linestyle='-', linewidth=2, label=f'正向绿波带 (带宽 {band_fwd:.1f}s)')
    ax.plot(L_line, t_fwd_upper, color='blue', linestyle='-', linewidth=2)
    ax.fill_between(L_line, t_fwd_lower, t_fwd_upper, color='#1f77b4', alpha=0.2)

    # 反向绿波 (注意坐标反转)
    L_rev_coords = np.array([0, positions[-1]])
    t_bwd_lower = L_rev_coords / v_bwd + max_k1_bwd
    t_bwd_upper = L_rev_coords / v_bwd + min_k2_bwd
    if t_bwd_lower[0] < 0:
        t_bwd_lower += cycle_time
        t_bwd_upper += cycle_time

    ax.plot(L_line, t_bwd_lower[::-1], color='orange', linestyle='-', linewidth=2,
            label=f'反向绿波带 (带宽 {band_bwd:.1f}s)')
    ax.plot(L_line, t_bwd_upper[::-1], color='orange', linestyle='-', linewidth=2)
    ax.fill_between(L_line, t_bwd_lower[::-1], t_bwd_upper[::-1], color='orange', alpha=0.15)

    ax.set_xlabel("距离 (米)")
    ax.set_ylabel("时间 (秒)")
    ax.set_title(f"双向干线绿波数解法时空图 (周期:{cycle_time}s)")
    ax.set_ylim(0, cycle_time * 1.5)
    ax.invert_yaxis()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    return {
        "cycle": cycle_time,
        "greens": green_times,
        "offsets_fwd": offsets_fwd,
        "offsets_bwd": offsets_bwd,
        "band_fwd": band_fwd,
        "band_bwd": band_bwd,
        "fig": fig
    }


def run_lvbo_agent(**kwargs):
    result = calculate_green_wave(kwargs)

    # 组装终极 Prompt 数据
    data_summary = f"""
    =================【预设固定参数】=================
    1. 交叉口数量：{len(kwargs['flows'])} 个
    2. 路段间距(m)：{kwargs['distances']}
    3. 协调车速：正向 {kwargs['v_fwd']} km/h，反向 {kwargs['v_bwd']} km/h
    4. 饱和流量(pcu/h)：{kwargs['sat_flow'] * kwargs['lanes']} 
    5. 最小/最大绿灯：{kwargs['min_green']}s / {kwargs['max_green']}s
    6. 黄灯/全红/损失：{kwargs['yellow_time']}s / {kwargs['all_red']}s / {kwargs['loss_time']}s
    7. 相位方案：{kwargs['phase_plan']} | 相序：{kwargs['phase_seq']}
    8. 周期约束：[{kwargs['min_cycle']}s, {kwargs['max_cycle']}s]
    9. 基准路口：路口 {kwargs['base_intersection']}
    10. 权重与带宽下限：正向权重 {kwargs['w_fwd']} | 反向权重 {kwargs['w_bwd']} | 带宽阈值 {kwargs['min_bandwidth']}s

    ===============【智能体优化变量结果】===============
    1. 优化后的公共周期 C (s)：{result['cycle']}
    2. 各交叉口绿灯分配 (s)：{result['greens']}
    3. 正向绝对相位差 (s)：{result['offsets_fwd']}
    4. 反向绝对相位差 (s)：{result['offsets_bwd']}
    5. 正向绿波带宽 (s)：{result['band_fwd']:.1f}
    6. 反向绿波带宽 (s)：{result['band_bwd']:.1f}
    """

    key = read_key()
    if not key:
        return result['fig'], "【系统提示】未找到 API Key，无法生成文字方案。"

    try:
        with open(SYS_PROMPT_PATH, encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "你是一个交通信号控制专家。"

    # 再次强化大模型对固定参数与优化变量的认知
    system_prompt += """

    【报告结构强制要求】
    请严格按照以下逻辑撰写报告，突出“约束条件”与“优化结果”的对比：

    一、 路网预设与约束条件概述
    （复述输入中的预设固定参数，说明这是不可更改的物理约束和硬性下限）

    二、 智能体优化变量与算法推演
    （重点写明智能体是如何在给定周期上下限、绿灯上下限内，计算出最终公共周期和绿灯时间的。并简单提及数解法理想间距推演。）

    三、 双向带宽截距分析
    （结合正向车速和反向车速，分析正向带宽和反向带宽的大小。如果某一向带宽低于了预设的最低阈值，请给出警醒。）

    四、 优化结果配时表
    （输出Markdown表格，包含绿灯、正向相位差、反向相位差）

    五、 实施建议与保障措施
    """

    client = DeepSeekClient(key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请基于以下严谨的数据，生成一份专家级的交通工程师绿波设计报告：\n{data_summary}"}
    ]

    try:
        text_plan = client.chat(messages)
    except Exception as e:
        text_plan = f"【请求大模型失败】{e}"

    return result['fig'], text_plan
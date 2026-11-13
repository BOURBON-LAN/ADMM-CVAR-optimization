import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from ADMM_optimizer import ADMM_Optimizer

# =========================================================
# 1. BCRP 求解器
# =========================================================
def find_bcrp_weights(realized_returns):
    """
    通过 SLSQP 数值优化寻找后视最优恒定再平衡组合 (BCRP)
    """
    T, N = realized_returns.shape
    def objective(P):
        portfolio_returns = np.dot(realized_returns, P)
        return -np.sum(np.log(portfolio_returns + 1e-15))
    
    constraints = ({'type': 'eq', 'fun': lambda P: np.sum(P) - 1.0})
    bounds = [(0, 1) for _ in range(N)]
    init_P = np.ones(N) / N
    res = minimize(objective, init_P, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-9)
    return res.x

# =========================================================
# 2. 核心回测引擎 (包含终端打印)
# =========================================================
def run_backtest_engine(file_path, label, test_time='2018-01-01', lookback=60, test_stock_num=100, gamma=0.01):
    # 1. 加载并截断数据
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df[df.index >= pd.to_datetime(test_time)].fillna(1.0)
    
    np.random.seed(42)
    selected_stocks = np.random.choice(df.columns.tolist(), test_stock_num, replace=False)
    data = df[selected_stocks].values
    
    # 2. 数据时间轴对齐 (跳过预热期)
    dates_idx = df.index[lookback+1:]
    realized_returns = data[lookback+1:]
    
    # 3. 计算 BCRP 基准
    print(f"\n" + "="*60)
    print(f"🚀 开始测试集: [{label}] | 周期: {dates_idx[0].date()} 至 {dates_idx[-1].date()}")
    print("正在计算全周期最优 BCRP 权重...")
    bcrp_weights = find_bcrp_weights(realized_returns)
    wealth_bcrp = np.cumprod([np.dot(bcrp_weights, r) for r in realized_returns])
    
    # 4. ADMM 动态策略运行与终端打印
    print(f"正在执行 ADMM 动态调仓策略 (Gamma = {gamma})...")
    print("-" * 60)
    print(f"{'Date':<15} | {'ADMM Wealth':<15} | {'Weight Std':<15}")
    print("-" * 60)
    
    optimizer = ADMM_Optimizer(N=test_stock_num, gamma=gamma, eta=0.01, lr=0.01)
    current_P = np.ones(test_stock_num) / test_stock_num
    wealth_admm = []
    weight_history = []
    
    w_acc = 1.0
    start_t = time.time()
    
    for t in range(lookback, len(data) - 1):
        # 求解权重
        current_P = optimizer.solve(data[t], data[t-lookback:t], current_P)
        weight_history.append(current_P.copy())
        
        # 计算策略累积财富
        r_next = data[t+1]
        w_acc *= np.dot(current_P, r_next)
        wealth_admm.append(w_acc)
        
        # 终端状态打印：每 200 个交易日打印一次
        if (t - lookback) % 200 == 0 or t == len(data) - 2:
            curr_date = df.index[t+1].date()
            curr_std = np.std(current_P)
            print(f"{str(curr_date):<15} | {w_acc:<15.4f} | {curr_std:<15.6f}")

    print("-" * 60)
    print(f"[{label}] 运行完毕! 耗时: {time.time() - start_t:.2f}s | 最终净值: ADMM={wealth_admm[-1]:.2f}, BCRP={wealth_bcrp[-1]:.2f}")
    
    return dates_idx, wealth_admm, wealth_bcrp, weight_history, realized_returns

# =========================================================
# 3. 目标函数遗憾值计算
# =========================================================
def calculate_regret(weights, returns, bcrp_wealth, gamma):
    """
    计算累计遗憾与平均遗憾
    公式: Regret = Sum(Log_BCRP) - [Sum(Log_ADMM) + gamma * Sum(L1_Cost)]
    """
    weights = np.array(weights)
    returns = np.array(returns)
    T = len(weights)
    
    sum_ln_bcrp = np.log(np.array(bcrp_wealth))
    
    admm_daily_ret = np.einsum('ij,ij->i', weights, returns)
    sum_ln_admm = np.cumsum(np.log(np.maximum(admm_daily_ret, 1e-12)))
    
    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    turnover = np.insert(turnover, 0, 0) # 补齐第一天的调仓(设为0)
    cumulative_turnover = np.cumsum(turnover)
    
    cumulative_regret = sum_ln_bcrp - (sum_ln_admm + gamma * cumulative_turnover)
    average_regret = cumulative_regret / np.arange(1, T + 1)
    
    return cumulative_regret, average_regret

# =========================================================
# 4. 可视化模块
# =========================================================
def plot_global_wealth(results_dict):
    """绘制所有数据集上的财富累积对比图 (一张图)"""
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#d62728'] # 为不同数据集分配主色调
    
    for i, (label, res) in enumerate(results_dict.items()):
        dates, admm_w, bcrp_w, _, _ = res
        c = colors[i % len(colors)]
        
        plt.plot(dates, admm_w, label=f'ADMM - {label}', lw=2, color=c)
        plt.plot(dates, bcrp_w, label=f'BCRP - {label}', lw=1.5, ls='--', color=c, alpha=0.6)
        
    plt.yscale('log')
    plt.title("Cumulative Wealth Comparison: ADMM vs BCRP (Log Scale)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Wealth", fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_dataset_regret(dates, cum_regret, avg_regret, label, gamma):
    """绘制单一数据集的遗憾值与次线性检验图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 累计遗憾
    ax1.plot(dates, cum_regret, color='crimson', lw=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"Objective Regret Analysis - {label} ($\gamma$={gamma})", fontsize=14)
    ax1.set_ylabel("Cumulative Regret ($R_T$)")
    ax1.grid(True, alpha=0.3)
    
    # 平均遗憾 (次线性检验)
    ax2.plot(dates, avg_regret, color='royalblue', lw=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Sublinear Convergence Test ($R_T / T$)", fontsize=12)
    ax2.set_ylabel("Average Regret")
    ax2.set_xlabel("Date")
    
    # 添加状态标签
    status = "Sublinear (Outperforming)" if avg_regret[-1] < 0 else "Converging"
    color = 'green' if avg_regret[-1] < 0 else 'orange'
    ax2.text(dates[len(dates)//4], np.min(avg_regret)*0.8 if np.min(avg_regret)<0 else np.max(avg_regret)*0.8, 
             f"Status: {status}", bbox=dict(facecolor='white', alpha=0.9), fontsize=12, color=color)
    
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# 5. 主程序流
# =========================================================
if __name__ == "__main__":
    # 配置
    datasets = {
        "Manufacturing": "申万制造成分股历史涨跌幅.csv",
        "Consumption": "申万消费成分股历史涨跌幅.csv"
    }
    GAMMA_PARAM = 0.01
    
    all_results = {}
    
    # 1. 运行所有回测
    for name, path in datasets.items():
        try:
            res = run_backtest_engine(path, name, test_time='2018-01-01', gamma=GAMMA_PARAM)
            all_results[name] = res
        except FileNotFoundError:
            print(f"⚠️ 跳过：未找到文件 {path}")
            
    if not all_results:
        print("没有可用的数据结果，程序退出。")
        exit()

    # 2. 画图 1: 两大集合并的净值曲线图
    plot_global_wealth(all_results)
    
    # 3. 画图 2 & 3: 各个数据集的遗憾值收敛图
    for name, res in all_results.items():
        dates, _, bcrp_w, weights, rets = res
        # 计算该数据集的遗憾值
        cum_regret, avg_regret = calculate_regret(weights, rets, bcrp_w, GAMMA_PARAM)
        # 画图
        plot_dataset_regret(dates, cum_regret, avg_regret, name, GAMMA_PARAM)
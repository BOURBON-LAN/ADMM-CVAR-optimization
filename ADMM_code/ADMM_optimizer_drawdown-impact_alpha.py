import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from ADMM_optimizer import ADMM_Optimizer

# --- BCRP 求解器 ---
def find_bcrp_weights(realized_returns):
    T, N = realized_returns.shape
    def objective(P):
        portfolio_returns = np.dot(realized_returns, P)
        return -np.sum(np.log(portfolio_returns + 1e-15))
    constraints = ({'type': 'eq', 'fun': lambda P: np.sum(P) - 1.0})
    bounds = [(0, 1) for _ in range(N)]
    init_P = np.ones(N) / N
    res = minimize(objective, init_P, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

# --- 核心回测函数 ---
def run_backtest_with_alpha(data_matrix, dates, alpha_val, gamma=0.01, lookback=60):
    N = data_matrix.shape[1]
    optimizer = ADMM_Optimizer(N=N, alpha=alpha_val, gamma=gamma, eta=0.01, lr=0.01)
    
    current_P = np.ones(N) / N
    wealth_history = []
    w_acc = 1.0
    
    for t in range(lookback, len(data_matrix) - 1):
        current_P = optimizer.solve(data_matrix[t], data_matrix[t-lookback:t], current_P)
        r_next = data_matrix[t+1]
        w_acc *= np.dot(current_P, r_next)
        wealth_history.append(w_acc)
        
    return wealth_history

# --- 计算动态回撤曲线 ---
def calculate_drawdown_curve(wealth_list):
    """返回每日的回撤比例序列 (正数表示跌幅)"""
    wealth = np.array(wealth_list)
    peak = np.maximum.accumulate(wealth) # 历史最高点
    drawdown = (peak - wealth) / peak
    return drawdown

# =========================================================
# 主程序：Alpha 影响实验与双图展示
# =========================================================
if __name__ == "__main__":
    file_path = "申万消费成分股历史涨跌幅.csv"
    START_DATE = '2018-01-01'
    ALPHA_LIST = [0.85, 0.90, 0.95, 0.99] # 对比不同风险偏好
    GAMMA = 0.01
    LOOKBACK = 60
    STOCK_NUM = 100 

    # 1. 加载数据
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df[df.index >= pd.to_datetime(START_DATE)].fillna(1.0)
    
    np.random.seed(42)
    selected_stocks = np.random.choice(df.columns.tolist(), STOCK_NUM, replace=False)
    data_matrix = df[selected_stocks].values
    dates_idx = df.index[LOOKBACK+1:]
    realized_returns = data_matrix[LOOKBACK+1:]

    # 2. 计算基准 BCRP
    print(f"正在计算 BCRP 基准...")
    bcrp_w = find_bcrp_weights(realized_returns)
    bcrp_wealth = np.cumprod([np.dot(bcrp_w, r) for r in realized_returns])
    bcrp_dd_curve = calculate_drawdown_curve(bcrp_wealth)

    # 3. 初始化画图 (上下两个子图)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    print("\n" + "="*60)
    print(f"{'Alpha':<10} | {'Final Wealth':<15} | {'Max Drawdown (%)':<15}")
    print("-" * 60)

    # 颜色库，方便上下两图颜色对应
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

    # 4. 循环测试不同 Alpha 并绘图
    for i, alpha in enumerate(ALPHA_LIST):
        # 运行回测
        admm_wealth = run_backtest_with_alpha(data_matrix, dates_idx, alpha, gamma=GAMMA, lookback=LOOKBACK)
        
        # 计算回撤曲线与最大回撤
        dd_curve = calculate_drawdown_curve(admm_wealth)
        mdd = np.max(dd_curve)
        print(f"{alpha:<10.2f} | {admm_wealth[-1]:<15.4f} | {mdd*100:<15.2f}%")
        
        # 绘制财富曲线 (上半图)
        ax1.plot(dates_idx, admm_wealth, label=f'ADMM ($\\alpha$={alpha})', color=colors[i], lw=1.5)
        
        # 绘制回撤曲线 (下半图，乘以-100转换为百分比负值，更符合直觉)
        ax2.fill_between(dates_idx, 0, -dd_curve * 100, label=f'ADMM ($\\alpha$={alpha})', color=colors[i], alpha=0.2)
        ax2.plot(dates_idx, -dd_curve * 100, color=colors[i], lw=1)

    # 5. 绘制 BCRP 对比基准
    ax1.plot(dates_idx, bcrp_wealth, label='BCRP (Benchmark)', color='black', linestyle='--', alpha=0.8, lw=2)
    ax2.plot(dates_idx, -bcrp_dd_curve * 100, label='BCRP (Benchmark)', color='black', linestyle='--', alpha=0.8, lw=2)

    # 6. 图表格式化设置
    # 上半图：累积财富
    ax1.set_yscale('log')
    ax1.set_title(f"Cumulative Wealth & Drawdown Impact by CVaR $\\alpha$ (since {START_DATE})", fontsize=14)
    ax1.set_ylabel("Cumulative Wealth (Log Scale)", fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)

    # 下半图：动态回撤
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    # 限制 y 轴上限为 0，因为回撤都是负数
    ax2.set_ylim(top=0) 

    plt.tight_layout()
    plt.show()
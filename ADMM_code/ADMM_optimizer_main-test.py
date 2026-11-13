import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from ADMM_optimizer import ADMM_Optimizer

# =========================================================
# 1. BCRP 求解器 (作为对比基准)
# =========================================================
def find_bcrp_weights(realized_returns):
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
# 2. 核心回测引擎 (增加详细持仓监控)
# =========================================================
def run_backtest_engine(file_path, label, test_time='2018-01-01', lookback=60, test_stock_num=100, gamma=0.0003):
    # 数据加载 (适配 2018-2026 长周期)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df[df.index >= pd.to_datetime(test_time)].fillna(1.0)
    
    np.random.seed(42)
    stock_names = np.random.choice(df.columns.tolist(), test_stock_num, replace=False)
    data = df[stock_names].values
    
    dates_idx = df.index[lookback+1:]
    realized_returns = data[lookback+1:]
    
    print(f"\n" + "="*90)
    print(f" 启动回测: [{label}] | Gamma={gamma} | 样本量: {len(data)}天")
    print(f"资产范围: {test_stock_num}只成分股 | 历史回看周期: {lookback}天")
    print("="*90)
    
    # BCRP 基准计算
    bcrp_weights = find_bcrp_weights(realized_returns)
    wealth_bcrp = np.cumprod([np.dot(bcrp_weights, r) for r in realized_returns])
    
    # ADMM 策略初始化
    optimizer = ADMM_Optimizer(N=test_stock_num, gamma=gamma, eta=0.01, lr=0.01)
    current_P = np.ones(test_stock_num) / test_stock_num
    wealth_admm = []
    
    # 终端打印表头
    print(f"{'Date':<12} | {'Wealth':<10} | {'Std':<8} | {'Top 5 Assets Allocation (ID: Weight%)'}")
    print("-" * 90)
    
    w_acc = 1.0
    # 严格因果回测循环
    for t in range(lookback, len(data) - 1):
        # 1. 在 t 时刻计算 t+1 的权重 (仅使用截止到 t 的数据)
        weight_for_tomorrow = optimizer.solve(data[t], data[t-lookback:t], current_P)
        
        # 2. 应用到 t+1 的实际收益
        r_tomorrow = data[t+1]
        w_acc *= np.dot(weight_for_tomorrow, r_tomorrow)
        wealth_admm.append(w_acc)
        
        # 3. 每 200 天打印一次资产配置明细
        if (t - lookback) % 200 == 0 or t == len(data) - 2:
            curr_date = df.index[t+1].date()
            curr_std = np.std(weight_for_tomorrow)
            
            # 获取权重最高的前 5 只股票
            top_indices = np.argsort(weight_for_tomorrow)[-5:][::-1]
            top_weights_str = " | ".join([f"#{idx}: {weight_for_tomorrow[idx]*100:5.2f}%" for idx in top_indices])
            
            print(f"{str(curr_date):<12} | {w_acc:<10.4f} | {curr_std:<8.5f} | {top_weights_str}")
            
        current_P = weight_for_tomorrow

    print("-" * 90)
    return dates_idx, wealth_admm, wealth_bcrp

# =========================================================
# 3. 性能评估与可视化
# =========================================================
def calculate_sharpe_ratio(wealth_sequence, risk_free_rate=0.03):
    wealth = np.array(wealth_sequence)
    daily_returns = np.diff(wealth) / wealth[:-1]
    if len(daily_returns) < 2: return 0.0
    return (np.mean(daily_returns) * 252 - risk_free_rate) / (np.std(daily_returns) * np.sqrt(252))

def plot_all_metrics(results_dict):
    # 1. 绘制净值曲线 (Log Scale)
    plt.figure(figsize=(12, 6))
    for name, (dates, admm_w, bcrp_w) in results_dict.items():
        plt.plot(dates, admm_w, label=f'ADMM - {name}', lw=2)
        plt.plot(dates, bcrp_w, label=f'BCRP - {name}', lw=1, ls='--', alpha=0.6)
    plt.yscale('log')
    plt.title("Cumulative Wealth (2018-2026, $\gamma=0.0003$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. 绘制夏普比率对比柱状图
    datasets = list(results_dict.keys())
    sharpe_admm = [calculate_sharpe_ratio(results_dict[n][1]) for n in datasets]
    sharpe_bcrp = [calculate_sharpe_ratio(results_dict[n][2]) for n in datasets]

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, sharpe_bcrp, width, label='BCRP', color='#7f7f7f', alpha=0.8, edgecolor='black')
    rects2 = ax.bar(x + width/2, sharpe_admm, width, label='ADMM-CVaR', color='#2ca02c', alpha=0.9, edgecolor='black')

    ax.set_ylabel('Annualized Sharpe Ratio')
    ax.set_title('Risk-Adjusted Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # 数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.show()

# =========================================================
# 4. 主程序
# =========================================================
if __name__ == "__main__":
    datasets_config = {
        "Manufacturing": "申万制造成分股历史涨跌幅.csv",
        "Consumption": "申万消费成分股历史涨跌幅.csv"
    }
    
    GAMMA_VAL = 0.0003  # 锁定 Gamma 参数
    all_results = {}

    for name, path in datasets_config.items():
        try:
            all_results[name] = run_backtest_engine(path, name, gamma=GAMMA_VAL)
        except Exception as e:
            print(f"跳过 {name}, 错误原因: {e}")

    if all_results:
        plot_all_metrics(all_results)
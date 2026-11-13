#gamma和alpha的不同组合的影响效果
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from ADMM_optimizer import ADMM_Optimizer # 确保与你的优化器在同级目录

# --- 核心回测函数 ---
def run_backtest_for_final_wealth(data_matrix, alpha_val, gamma_val, lookback=60):
    """只返回最终的累积财富，以加快网格搜索速度"""
    N = data_matrix.shape[1]
    # 初始化 ADMM 优化器
    optimizer = ADMM_Optimizer(N=N, alpha=alpha_val, gamma=gamma_val, eta=0.01, lr=0.01)
    
    current_P = np.ones(N) / N
    w_acc = 1.0
    
    for t in range(lookback, len(data_matrix) - 1):
        current_P = optimizer.solve(data_matrix[t], data_matrix[t-lookback:t], current_P)
        r_next = data_matrix[t+1]
        w_acc *= np.dot(current_P, r_next)
        
    return w_acc

# =========================================================
# 主程序：2D 网格搜索与绘图
# =========================================================
if __name__ == "__main__":
    file_path = "申万消费成分股历史涨跌幅.csv"
    START_DATE = '2018-01-01'
    LOOKBACK = 60
    STOCK_NUM = 100 # 为加快搜索速度，可适当减少股票池数量

    # 1. 定义要搜索的参数网格 (你可以根据需要调细或调粗)
    alphas = [0.80, 0.85, 0.90, 0.95, 0.99]
    gammas = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # 2. 加载数据
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df[df.index >= pd.to_datetime(START_DATE)].fillna(1.0)
        np.random.seed(42)
        selected_stocks = np.random.choice(df.columns.tolist(), STOCK_NUM, replace=False)
        data_matrix = df[selected_stocks].values
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    # 3. 执行网格搜索
    print(f"开始执行二维网格搜索，共需计算 {len(alphas) * len(gammas)} 次...")
    results_matrix = np.zeros((len(alphas), len(gammas)))

    for i, a in enumerate(alphas):
        for j, g in enumerate(gammas):
            print(f"Testing: alpha={a}, gamma={g}...", end=" ")
            start_t = time.time()
            final_wealth = run_backtest_for_final_wealth(data_matrix, a, g, LOOKBACK)
            results_matrix[i, j] = final_wealth
            print(f"Result: {final_wealth:.4f} (耗时: {time.time()-start_t:.2f}s)")

    # 构建 DataFrame 方便画图
    results_df = pd.DataFrame(results_matrix, index=alphas, columns=gammas)
    
    # ==========================================
    # 4. 绘制双图 (热力图 + 3D 曲面图)
    # ==========================================
    fig = plt.figure(figsize=(16, 6))

    # --- 子图 1: 热力图 (Heatmap) ---
    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(results_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1, cbar_kws={'label': 'Final Cumulative Wealth'})
    ax1.set_title("Heatmap of Final Wealth by $\\alpha$ and $\\gamma$", fontsize=14)
    ax1.set_xlabel("Gamma ($\\gamma$) - Transaction Penalty", fontsize=12)
    ax1.set_ylabel("Alpha ($\\alpha$) - CVaR Confidence", fontsize=12)
    ax1.invert_yaxis() # 让 y 轴从低到高排列

    # --- 子图 2: 3D 曲面图 (3D Surface) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(range(len(gammas)), range(len(alphas)))
    surf = ax2.plot_surface(X, Y, results_matrix, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # 设置 3D 图的轴标签
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels(gammas)
    ax2.set_yticks(range(len(alphas)))
    ax2.set_yticklabels(alphas)
    
    ax2.set_xlabel('Gamma ($\\gamma$)')
    ax2.set_ylabel('Alpha ($\\alpha$)')
    ax2.set_zlabel('Final Wealth')
    ax2.set_title("3D Surface of Parameter Landscape", fontsize=14)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Final Wealth')

    plt.tight_layout()
    plt.show()
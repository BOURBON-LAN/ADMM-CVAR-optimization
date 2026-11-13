from random import choice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ADMM_Optimizer:
    """通过内外部控制阈值的方法进行收敛计算的
    ADMM算法优化器"""
    def __init__(self, N=100,alpha=0.95,gamma=0.0003,eta=0.01,epsilon=0.001,
                 max_outer_iter=200, outer_tol=1e-4,
                 max_inter_iter=100, inter_tol=1e-5,lr=0.002):
        self.alpha = alpha  #CVaR的置信水平
        self.gamma = gamma  #L1正则化参数,即惩罚系数
        self.eta = eta
        self.epsilon = epsilon
        self.N = N  #资产数量
        # 外部迭代参数
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        
        #内部迭代参数（P-子问题）
        self.max_inter_iter = max_inter_iter
        self.inter_tol = inter_tol
        self.lr = lr
        
    def get_smooth_cvar_grad(self,P,X_history):
        T,N = X_history.shape
        losses = -X_history @ P
        #计算VaR
        var_alpha = np.quantile(losses,self.alpha)
        t = losses - var_alpha
        #进行平滑处理
        phi_prime = np.zeros_like(t)
        mask_mid = (t>0) & (t<=self.epsilon)
        mask_high = (t>self.epsilon)
        phi_prime[mask_mid] = t[mask_mid]/self.epsilon
        phi_prime[mask_high] = 1.0
        #计算最终梯度
        grad_cvar = 1 /(T*(1-self.alpha)) * (phi_prime @ X_history)
        return grad_cvar
    
    #开始求解问题
    def solve(self,X_last,X_history,P_prev):
        N = len(X_last)
        P = P_prev.copy()
        #初始化辅助变量和对偶变量
        z = np.zeros(N)
        s = P_prev.copy()
        #对lambda进行拆分
        l0 = 0
        l1 = np.zeros(N)
        l2 = np.zeros(N)    
        
        for k in range(self.max_outer_iter):
            #更新s-子问题
            s = np.maximum(P + (l2 / self.eta), 0)
            #更新z-子问题
            a_z = P - P_prev + (l1 / self.eta)
            z = np.sign(a_z) * np.maximum(np.abs(a_z) - (self.gamma / self.eta), 0)
            #更新P-子问题
            for i in range(self.max_inter_iter):
                P_inner_old = P.copy()
                #计算总体梯度
                g_cvar = self.get_smooth_cvar_grad(P,X_history)
                g_ret = -1.0*X_last / (P @ X_last+1e-9)  #防止除零导致梯度爆炸
                #ADMM罚项产生的梯度
                err0 = np.sum(P) - 1 + l0 / self.eta
                err1 = P - z - P_prev + l1 / self.eta
                err2 = P - s + l2 / self.eta
                g_admm = self.eta * (np.ones(N) * err0 + err1 + err2)
                #梯度更新
                P = P - self.lr * (g_cvar + g_ret + g_admm)
                #投影到单纯形
                P = np.maximum(P, 0)
                P /= (np.sum(P) + 1e-12)
                
                #检查内部迭代收敛，检查P的位移范数
                inner_diff = np.linalg.norm(P - P_inner_old)
                if inner_diff < self.inter_tol:
                    break
        
            #更新对偶变量
            r0 = np.sum(P) - 1
            r1 = P - z - P_prev
            r2 = P - s
        
            l0 += self.eta * r0
            l1 += self.eta * r1
            l2 += self.eta * r2
            #检查外部迭代收敛
            primal_residual = np.linalg.norm(np.concatenate([[r0], r1, r2]))
            if primal_residual < self.outer_tol:
                break
        
        
        return P

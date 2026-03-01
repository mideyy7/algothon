import numpy as np
import matplotlib.pyplot as plt
import os

# Create an output directory for the graphs
out_dir = "submission_graphs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def generate_ols_pair_trading_plot():
    """Generates a plot showing how the Rolling OLS Pair Trading works."""
    np.random.seed(42)
    # Simulate a cointegrated pair
    n = 1000
    t = np.arange(n)
    trend = np.sin(t / 50.0) * 10
    noise_A = np.random.normal(0, 1, n)
    noise_B = np.random.normal(0, 1, n)
    
    # Asset A is the driver
    price_A = 100 + trend + np.cumsum(noise_A)
    # Asset B is tied to A but drifts, then reverts
    price_B = 50 + 0.5 * price_A + np.sin(t / 20.0) * 5 + np.random.normal(0, 2, n)
    
    # Simulate the Rolling OLS Residuals
    window = 100
    residuals = []
    z_scores = []
    
    for i in range(window, n):
        x = price_A[i-window:i]
        y = price_B[i-window:i]
        
        mx = x.mean()
        my = y.mean()
        vx = ((x - mx)**2).mean() + 1e-12
        cov = ((x - mx) * (y - my)).mean()
        
        beta = cov / vx
        alpha = my - beta * mx
        
        fair_B = alpha + beta * price_A[i]
        res = price_B[i] - fair_B
        residuals.append(res)
        
    residuals = np.array(residuals)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    z_scores = (residuals - mean_res) / std_res
    
    # Plot 1: Prices & Spread
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, price_A, label="Leg A (e.g., LHR_COUNT)", color='blue', alpha=0.7)
    plt.plot(t, price_B * 2, label="Leg B Scaled (e.g., LHR_INDEX)", color='orange', alpha=0.9)
    plt.title("Rolling OLS Pair Trading Model: Asset Tracking")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    t_res = t[window:]
    plt.plot(t_res, z_scores, color='purple', label="Residual Z-Score")
    plt.axhline(2.2, color='red', linestyle='--', label="Sell B, Buy A (z=2.2)")
    plt.axhline(-2.2, color='green', linestyle='--', label="Buy B, Sell A (z=-2.2)")
    plt.axhline(0.6, color='black', label="Flatten Exit (z=0.6)", alpha=0.5)
    plt.axhline(-0.6, color='black', alpha=0.5)
    
    # Highlight trades
    sell_signals = t_res[z_scores > 2.2]
    buy_signals = t_res[z_scores < -2.2]
    plt.scatter(sell_signals, z_scores[z_scores > 2.2], color='red', marker='v')
    plt.scatter(buy_signals, z_scores[z_scores < -2.2], color='green', marker='^')
    
    plt.title("Dynamic Alpha/Beta Spread Z-Score Execution")
    plt.xlabel("Ticks")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ols_pair_trading.png"), dpi=300)
    plt.close()
    print("Saved ols_pair_trading.png")

def generate_ewma_dip_buyer_plot():
    """Generates a plot showing the UpOnlyDipBuyer behavior."""
    np.random.seed(101)
    n = 500
    t = np.arange(n)
    
    # Asset trending up with sudden dips
    trend = t * 0.05
    noise = np.random.normal(0, 1.5, n)
    # Add severe dips
    noise[100:110] -= 10
    noise[250:260] -= 12
    noise[400:410] -= 8
    
    price = 100 + trend + noise
    
    # EWMA tracker
    alpha = 0.06
    ema = []
    v = price[0]
    for p in price:
        v = (1 - alpha) * v + alpha * p
        ema.append(v)
    ema = np.array(ema)
    
    diff = price - ema
    std = np.std(diff)
    z = diff / std
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, price, label="Live Ask Price", color='#1f77b4', linewidth=1.5)
    plt.plot(t, ema, label=r"EWMA Baseline ($\alpha=0.06$)", color='orange', linestyle='--', linewidth=2)
    
    buy_signals = t[z < -1.6]
    plt.scatter(buy_signals, price[buy_signals], color='green', s=80, marker='^', zorder=5, label="Buy Execution (Z < -1.6)")
    
    take_profit = t[(z > -0.3) & (np.roll(z < -1.6, 10))] # Approximation for visualization
    
    plt.title("UpOnlyDipBuyer: Asymmetrical Volatility Targeting")
    plt.ylabel("Price")
    plt.xlabel("Ticks")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ewma_dip_buyer.png"), dpi=300)
    plt.close()
    print("Saved ewma_dip_buyer.png")

if __name__ == "__main__":
    print("Generating statistical modeling charts...")
    generate_ols_pair_trading_plot()
    generate_ewma_dip_buyer_plot()
    print("Done!")

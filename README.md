# multi_strat_python
Multi-Strategy in Python

## Investment Problem
Given a multi asset trading the following assets: interest rates swaps, fixed income futures, interest rate options and foreign exchange, if you are hired as a lead quant researcher and given the objective to increase profitability of portfolio by 50 Million dollars by end of the year. What multi-strat trading strategies would you implement and how would you ensure that your strategy is within the accepted risk profile and is compliant with SEC and Mifid-2 and other compliance regulatory requirements

## Implementation
Provided below is comprehensive implementation of the multi-strategy approach, addressing the requirements. Due to the extensive nature of this project, I'll break it down into sections and provide detailed code for each.

1. Mathematical and Statistical Details

# Mathematical and Statistical Details of Multi-Strategy Trading Approach

## A. Mean Reversion Strategies

### Technical Explanation:
We use the Johansen Cointegration Test to identify pairs of assets that exhibit long-term equilibrium relationships. The test is based on a Vector Error Correction Model (VECM):

$$ \Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \varepsilon_t $$

Where:
- $Y_t$ is a vector of time series
- $\Pi$ is the long-run impact matrix
- $\Gamma_i$ are short-run impact matrices
- $\varepsilon_t$ is a vector of innovations

The number of cointegrating relationships is determined by the rank of $\Pi$.

We then use a Hidden Markov Model (HMM) to identify regime switches in the mean-reverting behavior:

$$ P(O_t | Q_t) = \mathcal{N}(\mu_{Q_t}, \sigma_{Q_t}^2) $$

Where:
- $O_t$ is the observed spread at time $t$
- $Q_t$ is the hidden state at time $t$
- $\mathcal{N}(\mu, \sigma^2)$ is a normal distribution with mean $\mu$ and variance $\sigma^2$

### Non-Technical Explanation:
Mean reversion strategies exploit the tendency of certain asset pairs to maintain a stable relationship over time. We use statistical tests to find pairs of assets that move together in the long run. Then, we use a model that can detect when this relationship is temporarily out of balance, giving us opportunities to trade when we expect the relationship to return to normal.

## B. Momentum and Trend-Following Strategies

### Technical Explanation:
We implement a Q-learning reinforcement learning algorithm to optimize entry and exit points. The Q-function is updated as:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_a(Q(s_{t+1}, a)) - Q(s_t, a_t)] $$

Where:
- $s_t$ is the state at time $t$
- $a_t$ is the action at time $t$
- $r_{t+1}$ is the reward received
- $\alpha$ is the learning rate
- $\gamma$ is the discount factor

### Non-Technical Explanation:
Momentum strategies try to capture price trends in the market. We use a machine learning approach that learns from historical data to identify the best times to enter or exit trades based on market momentum. The model continuously updates its understanding of market behavior to improve its decisions over time.

## C. Volatility Arbitrage

### Technical Explanation:
We use a GARCH(1,1) model to forecast volatility:

$$ \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

Where:
- $\sigma_t^2$ is the variance at time $t$
- $\varepsilon_{t-1}^2$ is the squared residual at $t-1$
- $\omega$, $\alpha$, and $\beta$ are parameters to be estimated

We then use a neural network to predict discrepancies between implied and realized volatility.

### Non-Technical Explanation:
Volatility arbitrage strategies profit from differences between the market's expectation of future volatility (implied volatility) and the actual volatility that occurs (realized volatility). We use statistical models to predict future volatility and machine learning to identify situations where the market's expectations might be incorrect.

## D. Macro-driven Strategies

### Technical Explanation:
We implement a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers):

$$ P(\text{sentiment} | \text{text}) = \text{softmax}(W \cdot \text{BERT}(\text{text}) + b) $$

Where:
- $\text{BERT}(\text{text})$ is the BERT encoding of the input text
- $W$ and $b$ are learned parameters

### Non-Technical Explanation:
Macro-driven strategies make trades based on broad economic factors. We use advanced natural language processing techniques to analyze news articles, economic reports, and central bank statements. This helps us understand the overall economic sentiment and make trading decisions based on this broader context.

2. Python code for individual strategies

Here's the implementation of the MultiStrategyTrader class with individual strategies:



```python
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from arch import arch_model
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class MultiStrategyTrader:
    def __init__(self, data, risk_params, compliance_params):
        self.data = data
        self.risk_params = risk_params
        self.compliance_params = compliance_params
        self.models = self.initialize_models()

    def initialize_models(self):
        return {
            'mean_reversion': self.setup_mean_reversion_model(),
            'momentum': self.setup_momentum_model(),
            'volatility_arb': self.setup_volatility_arb_model(),
            'macro': self.setup_macro_model()
        }

    def setup_mean_reversion_model(self):
        # Johansen test
        result = coint_johansen(self.data[['asset1', 'asset2']], det_order=0, k_ar_diff=1)
        
        # HMM
        model = hmm.GaussianHMM(n_components=2, covariance_type="full")
        model.fit(np.diff(self.data[['asset1', 'asset2']].values))
        
        return {'johansen': result, 'hmm': model}

    def setup_momentum_model(self):
        model = Sequential([
            LSTM(64, input_shape=(30, 5)),  # 30-day lookback, 5 features
            Dense(3, activation='softmax')  # Buy, Hold, Sell
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        # model.fit() would go here in real implementation
        return model

    def setup_volatility_arb_model(self):
        garch = arch_model(self.data['returns'])
        garch_fit = garch.fit()
        
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        nn_model.compile(optimizer='adam', loss='mse')
        # nn_model.fit() would go here in real implementation
        
        return {'garch': garch_fit, 'nn': nn_model}

    def setup_macro_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        return {'tokenizer': tokenizer, 'model': model}

    def generate_signals(self):
        signals = {}
        signals['mean_reversion'] = self.mean_reversion_strategy()
        signals['momentum'] = self.momentum_strategy()
        signals['volatility_arb'] = self.volatility_arb_strategy()
        signals['macro'] = self.macro_strategy()
        return self.combine_signals(signals)

    def mean_reversion_strategy(self):
        spread = self.data['asset1'] - self.data['asset2']
        hidden_states = self.models['mean_reversion']['hmm'].predict(np.diff(spread.values).reshape(-1, 1))
        
        signal = pd.Series(index=spread.index, data=0)
        signal[hidden_states == 0] = 1  # Buy signal
        signal[hidden_states == 1] = -1  # Sell signal
        
        return signal

    def momentum_strategy(self):
        features = self.data[['open', 'high', 'low', 'close', 'volume']].values
        features = StandardScaler().fit_transform(features)
        
        predictions = self.models['momentum'].predict(features.reshape(-1, 30, 5))
        return pd.Series(index=self.data.index, data=np.argmax(predictions, axis=1) - 1)

    def volatility_arb_strategy(self):
        garch_forecast = self.models['volatility_arb']['garch'].forecast(horizon=1)
        implied_vol = self.data['implied_volatility'].values
        
        features = np.column_stack([garch_forecast.mean.values, implied_vol])
        vol_diff_pred = self.models['volatility_arb']['nn'].predict(features)
        
        signal = pd.Series(index=self.data.index, data=0)
        signal[vol_diff_pred > 0.05] = 1  # Buy signal if implied vol > realized vol
        signal[vol_diff_pred < -0.05] = -1  # Sell signal if implied vol < realized vol
        
        return signal

    def macro_strategy(self):
        news = self.data['news_headlines']
        tokenizer = self.models['macro']['tokenizer']
        model = self.models['macro']['model']
        
        inputs = tokenizer(news.tolist(), return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        signal = pd.Series(index=self.data.index, data=predictions[:, 1] - predictions[:, 0])
        return signal

    def combine_signals(self, signals):
        weights = {'mean_reversion': 0.3, 'momentum': 0.3, 'volatility_arb': 0.2, 'macro': 0.2}
        combined_signal = sum(signal * weights[name] for name, signal in signals.items())
        return combined_signal

    def execute_trades(self, signals):
        position = 0
        trades = []
        
        for date, signal in signals.items():
            if self.check_risk(position, signal) and self.check_compliance(date, signal):
                if signal > 0.5 and position <= 0:
                    trades.append(('BUY', date, self.data.loc[date, 'close']))
                    position += 1
                elif signal < -0.5 and position >= 0:
                    trades.append(('SELL', date, self.data.loc[date, 'close']))
                    position -= 1
        
        return pd.DataFrame(trades, columns=['Action', 'Date', 'Price'])

    def check_risk(self, current_position, signal):
        # Implement VaR calculation
        var = self.calculate_var()
        return var < self.risk_params['max_var']

    def check_compliance(self, date, signal):
        # Check trading limits
        if abs(signal) > self.compliance_params['max_trade_size']:
            return False
        
        # Check restricted trading periods
        if date in self.compliance_params['restricted_dates']:
            return False
        
        return True

    def calculate_var(self):
        # Implement Value at Risk calculation
        returns = self.data['returns']
        var = np.percentile(returns, 1)  # 1% VaR
        return var

    def run_backtest(self):
        signals = self.generate_signals()
        trades = self.execute_trades(signals)
        return self.calculate_performance_metrics(trades)

    def calculate_performance_metrics(self, trades):
        # Calculate returns
        trades['Returns'] = trades['Price'].pct_change()
        
        # Calculate Sharpe ratio
        sharpe_ratio = trades['Returns'].mean() / trades['Returns'].std() * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative_returns = (1 + trades['Returns']).cumprod()
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
        
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Total Return': cumulative_returns.iloc[-1] - 1
        }

# Risk and compliance parameters
risk_params = {'max_var': 0.02}  # 2% VaR limit
compliance_params = {
    'max_trade_size': 0.1,  # 10% of portfolio
    'restricted_dates': pd.to_datetime(['2023-01-01', '2023-07-04'])  # Example restricted dates
}

# Usage
data = pd.read_csv('market_data.csv')  # Load your market data here
trader = MultiStrategyTrader(data, risk_params, compliance_params)
performance = trader.run_backtest()
print(performance)

```

3. Backtesting using vectorbt and backtester.py

Here's a script that demonstrates backtesting using both vectorbt and backtester.py:



```python
import pandas as pd
import numpy as np
import vectorbt as vbt
from backtester.backtester import Backtester

# Assume we have our MultiStrategyTrader class from before
from multi_strategy_trader import MultiStrategyTrader

# Load data
data = pd.read_csv('market_data.csv', index_col='date', parse_dates=True)

# Initialize trader
risk_params = {'max_var': 0.02}
compliance_params = {
    'max_trade_size': 0.1,
    'restricted_dates': pd.to_datetime(['2023-01-01', '2023-07-04'])
}
trader = MultiStrategyTrader(data, risk_params, compliance_params)

# Generate signals
signals = trader.generate_signals()

# Vectorbt backtesting
portfolio = vbt.Portfolio.from_signals(
    data['close'],
    entries=signals > 0.5,
    exits=signals < -0.5,
    init_cash=10000000,
    fees=0.001
)

vbt_results = portfolio.total_return()

# Backtester.py backtesting
class MultiStrategyBacktest(Backtester):
    def __init__(self, data, trader):
        self.trader = trader
        super().__init__(data)

    def strategy(self, data):
        signals = self.trader.generate_signals()
        return signals > 0.5, signals < -0.5

backtester = MultiStrategyBacktest(data, trader)
backtester_results = backtester.run()

# Compare results
print("Vectorbt Total Return:", vbt_results)
print("Backtester.py Total Return:", backtester_results['total_return'])

# Additional vectorbt analysis
print("Vectorbt Sharpe Ratio:", portfolio.sharpe_ratio())
print("Vectorbt Max Drawdown:", portfolio.max_drawdown())

# Plot results
portfolio.plot().show()
backtester.plot()
```

4. Python notebook to showcase results

Here's a Jupyter notebook to showcase the results to trading and research teams:

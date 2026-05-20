# PyFinance vendor code

This directory stores reusable legacy code copied from:

`/Users/chenjunming/Desktop/KevinGit/PyFinance`

It is kept as vendor/reference code for the DRL portfolio-management experiment platform.

## Directory layout

- `MetricsFactory/`: original PyFinance indicator factory.
- `GetData/`: original Tushare/AkShare data download and preprocessing helpers.
- `AutoFactorCreator/`: original automatic factor engineering scripts.
- `legacy_rl_scripts/`: original DQN, PPO, CNN-PPO, EIIE prototype scripts used as algorithm references.
- `main_cal_metrics.py`: original MetricsFactory entry script.
- `main_get_data.py`: original data download/preprocess entry script.
- `set_tushare.py`: original Tushare setup file.

## DRL_PM integration

Use `/Users/chenjunming/Desktop/DRL_PM/scripts/run_pyfinance_metrics_factory.py` to compute MetricsFactory indicators from the DRL_PM parquet data files. That wrapper imports this local vendor copy and converts pandas 3 arrays to writable C-contiguous NumPy arrays before calling numba-backed rolling indicators.

Use `/Users/chenjunming/Desktop/DRL_PM/scripts/rebuild_etf_lof_data.py` to rebuild the ETF/LOF raw and processed parquet data without relying on Tushare token access.

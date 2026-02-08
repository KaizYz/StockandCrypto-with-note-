# Execution Assumptions

- scope: paper_trading_only
- fill_mode: next_open
- delay_bars: 1
- cost_components_bps:
  - fee_bps: 10.0
  - slippage_bps: 10.0
  - impact_bps_proxy: 1.0
- impact_model: lambda_bps=1.0, beta=0.5
- kill_switch: enabled=True, required_health_checks=3, trial_scale=0.25, trial_windows=1, admin_role=ops_admin

## Notes
- next_open/vwap/mid are supported in contract; current paper runtime falls back to latest-price proxy when tick/orderbook is unavailable.
- This file is generated during reporting/export and should be archived with release_signoff.

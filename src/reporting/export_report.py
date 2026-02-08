from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config, save_yaml
from src.utils.io import load_json, save_json, write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _load_thresholds(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    report_cfg = cfg.get("reporting", {})
    thresholds_file = str(report_cfg.get("thresholds_file", "configs/go_live_thresholds.yaml"))
    path = Path(thresholds_file)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent.parent / path
    if path.exists():
        return load_config(path)
    return {}


def _eval_metric(value: float, *, op: str, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return abs(value - threshold) <= 1e-12
    return False


def _file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _composite_hash(paths: List[Path]) -> str:
    h = hashlib.sha256()
    added = 0
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        h.update(str(p).encode("utf-8"))
        h.update(_file_hash(p).encode("utf-8"))
        added += 1
    return h.hexdigest() if added > 0 else ""


def _config_hash(cfg: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(cfg or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return ""


def _load_runtime_tables(processed_dir: Path) -> Dict[str, Any]:
    backtest_dir = processed_dir / "backtest"
    execution_dir = processed_dir / "execution"
    data_integrity_path = processed_dir / "data_integrity_checks.json"
    drift_path = processed_dir / "drift_monitor_daily.csv"
    calendar_path = processed_dir / "calendar_alignment_report.json"
    metrics_summary_path = backtest_dir / "metrics_summary.csv"
    metrics_by_fold_path = backtest_dir / "metrics_by_fold.csv"
    trades_path = backtest_dir / "trades.csv"
    orders_path = execution_dir / "paper_orders.csv"

    return {
        "data_integrity": load_json(data_integrity_path) if data_integrity_path.exists() else {"summary": {}},
        "drift_df": pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame(),
        "calendar_report": load_json(calendar_path) if calendar_path.exists() else {},
        "metrics_summary": pd.read_csv(metrics_summary_path) if metrics_summary_path.exists() else pd.DataFrame(),
        "metrics_by_fold": pd.read_csv(metrics_by_fold_path) if metrics_by_fold_path.exists() else pd.DataFrame(),
        "trades_df": pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame(),
        "orders_df": pd.read_csv(orders_path) if orders_path.exists() else pd.DataFrame(),
    }


def _execution_fail_rate(orders_df: pd.DataFrame) -> float:
    if orders_df.empty or "status" not in orders_df.columns:
        return float("nan")
    status = orders_df["status"].astype(str).str.lower()
    failures = status.isin({"error", "failed", "rejected", "cancelled"}).sum()
    total = len(status)
    return float(failures / total) if total > 0 else float("nan")


def _build_market_context(
    *,
    metrics_summary: pd.DataFrame,
    metrics_by_fold: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    ctx: Dict[str, Dict[str, float]] = {}
    policy_summary = (
        metrics_summary[metrics_summary["strategy"].astype(str) == "policy"].copy()
        if not metrics_summary.empty and "strategy" in metrics_summary.columns
        else pd.DataFrame()
    )
    for market in sorted(policy_summary["market"].astype(str).unique()) if not policy_summary.empty else []:
        row = policy_summary[policy_summary["market"].astype(str) == str(market)].mean(numeric_only=True)
        market_metrics: Dict[str, float] = {}
        for c in ["total_return", "cagr", "max_drawdown", "sharpe", "sortino", "volatility", "win_rate", "avg_win_loss_ratio", "profit_factor", "expectancy", "trades_count"]:
            market_metrics[c] = _safe_float(row.get(c))

        trades_m = pd.DataFrame()
        if not trades_df.empty and {"market", "strategy"}.issubset(trades_df.columns):
            trades_m = trades_df[
                (trades_df["market"].astype(str) == str(market))
                & (trades_df["strategy"].astype(str) == "policy")
            ].copy()
        market_metrics["min_trades_in_window"] = float(len(trades_m))

        by_fold_m = pd.DataFrame()
        if not metrics_by_fold.empty and {"market", "strategy"}.issubset(metrics_by_fold.columns):
            by_fold_m = metrics_by_fold[
                (metrics_by_fold["market"].astype(str) == str(market))
                & (metrics_by_fold["strategy"].astype(str) == "policy")
            ].copy()
        for col in ["sharpe", "total_return", "max_drawdown"]:
            market_metrics[f"{col}_std"] = _safe_float(pd.to_numeric(by_fold_m.get(col), errors="coerce").std(ddof=0)) if not by_fold_m.empty else float("nan")

        ctx[str(market)] = market_metrics
    return ctx


def _evaluate_go_live(
    *,
    thresholds: Dict[str, Any],
    data_integrity: Dict[str, Any],
    drift_df: pd.DataFrame,
    calendar_report: Dict[str, Any],
    metrics_summary: pd.DataFrame,
    metrics_by_fold: pd.DataFrame,
    trades_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> Dict[str, Any]:
    rules = thresholds.get("go_live_rules", {})
    global_rules = rules.get("global", [])
    market_rules = rules.get("markets", {})

    pass_rate = _safe_float(data_integrity.get("summary", {}).get("pass_rate"))
    drift_red_count = float((drift_df.get("alert_level", pd.Series(dtype=str)).astype(str) == "red").sum()) if not drift_df.empty else 0.0
    execution_fail_rate = _execution_fail_rate(orders_df)
    calendar_pass = 1.0 if _safe_bool(calendar_report.get("all_pass")) else 0.0

    global_metrics = {
        "data_pass_rate": pass_rate,
        "drift_red_count": drift_red_count,
        "execution_fail_rate": execution_fail_rate,
        "calendar_alignment_pass": calendar_pass,
    }
    market_ctx = _build_market_context(
        metrics_summary=metrics_summary,
        metrics_by_fold=metrics_by_fold,
        trades_df=trades_df,
    )

    results: List[Dict[str, Any]] = []
    for rule in global_rules:
        metric = str(rule.get("metric"))
        op = str(rule.get("op", ">="))
        threshold = _safe_float(rule.get("value"))
        val = _safe_float(global_metrics.get(metric))
        ok = _eval_metric(val, op=op, threshold=threshold)
        results.append(
            {
                "scope": "global",
                "market": "all",
                "metric": metric,
                "value": val,
                "op": op,
                "threshold": threshold,
                "pass": ok,
                "note": "" if np.isfinite(val) else "metric_unavailable",
            }
        )

    for market, rules_list in market_rules.items():
        mvals = market_ctx.get(str(market), {})
        for rule in rules_list:
            metric = str(rule.get("metric"))
            op = str(rule.get("op", ">="))
            threshold = _safe_float(rule.get("value"))
            val = _safe_float(mvals.get(metric))
            ok = _eval_metric(val, op=op, threshold=threshold)
            results.append(
                {
                    "scope": "market",
                    "market": str(market),
                    "metric": metric,
                    "value": val,
                    "op": op,
                    "threshold": threshold,
                    "pass": ok,
                    "note": "" if np.isfinite(val) else "missing_market_metric",
                }
            )

    result_df = pd.DataFrame(results)
    all_pass = bool(result_df["pass"].all()) if not result_df.empty else False
    market_status = (
        result_df[result_df["scope"] == "market"]
        .groupby("market", as_index=False)["pass"]
        .all()
        .rename(columns={"pass": "market_pass"})
    ) if not result_df.empty else pd.DataFrame(columns=["market", "market_pass"])
    return {
        "all_pass": all_pass,
        "rules": results,
        "market_status": market_status.to_dict(orient="records"),
        "global_metrics": global_metrics,
        "market_metrics": market_ctx,
    }


def _render_report_markdown(
    *,
    go_live: Dict[str, Any],
    data_integrity: Dict[str, Any],
    metrics_summary: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> str:
    lines = [
        "# Training Readiness Report",
        "",
        f"- Overall decision: **{'GO' if go_live.get('all_pass') else 'NO-GO'}**",
        f"- Generated at (UTC): {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Global Metrics",
    ]
    for k, v in (go_live.get("global_metrics") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Data Gate", f"- Pass rate: {data_integrity.get('summary', {}).get('pass_rate', 'n/a')}"])

    lines.extend(["", "## Market Status"])
    for row in go_live.get("market_status", []):
        lines.append(f"- {row.get('market', 'unknown')}: {'PASS' if row.get('market_pass') else 'FAIL'}")

    lines.extend(["", "## Rules Detail"])
    for row in go_live.get("rules", []):
        lines.append(
            f"- [{row.get('scope')}/{row.get('market')}] {row.get('metric')} "
            f"{row.get('op')} {row.get('threshold')} | value={row.get('value')} "
            f"=> {'PASS' if row.get('pass') else 'FAIL'}"
        )

    lines.extend(["", "## Policy Metrics (summary)"])
    if metrics_summary.empty:
        lines.append("- no metrics_summary.csv available")
    else:
        policy = metrics_summary[metrics_summary["strategy"].astype(str) == "policy"].copy()
        if policy.empty:
            lines.append("- no policy rows found")
        else:
            cols = ["market", "symbol", "total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "trades_count"]
            cols = [c for c in cols if c in policy.columns]
            lines.append("")
            lines.append(policy[cols].to_markdown(index=False))

    lines.extend(["", "## Drift Alerts"])
    if drift_df.empty:
        lines.append("- no drift monitor data")
    else:
        top = drift_df.head(20)
        cols = [c for c in ["market", "symbol", "strategy", "alert_level", "reason"] if c in top.columns]
        lines.append("")
        lines.append(top[cols].to_markdown(index=False))
    return "\n".join(lines)


def _render_release_signoff(
    *,
    go_live: Dict[str, Any],
    release_manifest: Dict[str, Any],
) -> str:
    decision = "GO" if go_live.get("all_pass") else "NO-GO"
    lines = [
        "# Release Signoff",
        "",
        f"- Decision: **{decision}**",
        f"- Generated at (UTC): {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Release Manifest Snapshot",
        f"- release_name: {release_manifest.get('release_name', '-')}",
        f"- git_commit: {release_manifest.get('versions', {}).get('git_commit', '-')}",
        f"- config_hash: {release_manifest.get('versions', {}).get('config_hash', '-')}",
        f"- data_hash: {release_manifest.get('versions', {}).get('data_hash', '-')}",
        f"- model_version: {release_manifest.get('versions', {}).get('model_version', '-')}",
        "",
        "## Threshold Results",
    ]
    rules = pd.DataFrame(go_live.get("rules", []))
    if rules.empty:
        lines.append("- no threshold results")
    else:
        show = rules.copy()
        show["result"] = np.where(show["pass"].astype(bool), "PASS", "FAIL")
        cols = ["scope", "market", "metric", "value", "op", "threshold", "result", "note"]
        cols = [c for c in cols if c in show.columns]
        lines.append("")
        lines.append(show[cols].to_markdown(index=False))
    return "\n".join(lines)


def _infer_model_version(processed_dir: Path) -> str:
    candidates = [
        processed_dir / "predictions_latest_summary.json",
        processed_dir / "market_snapshot.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            payload = load_json(p)
            if isinstance(payload, dict):
                if "prediction_method" in payload:
                    return str(payload.get("prediction_method"))
                rows = payload.get("rows")
                if isinstance(rows, list) and rows:
                    method = rows[0].get("prediction_method")
                    if method:
                        return str(method)
        except Exception:
            continue
    return "baseline_momentum_quantile"


def _build_release_manifest(
    *,
    cfg: Dict[str, Any],
    config_path: str,
    processed_dir: Path,
    root_dir: Path,
) -> Dict[str, Any]:
    backtest_dir = processed_dir / "backtest"
    universe_path = backtest_dir / "universe_snapshot.json"
    universe_version = ""
    if universe_path.exists():
        universe_version = _file_hash(universe_path)[:16]

    data_hash = _composite_hash(
        [
            processed_dir / "features_hourly.csv",
            processed_dir / "features_daily.csv",
            processed_dir / "predictions_hourly.csv",
            processed_dir / "predictions_daily.csv",
            backtest_dir / "metrics_summary.csv",
            backtest_dir / "metrics_by_fold.csv",
        ]
    )
    cfg_hash = _config_hash(cfg)
    model_version = _infer_model_version(processed_dir)
    bt_cfg = cfg.get("backtest_multi_market", {}) if isinstance(cfg, dict) else {}
    policy_cfg = cfg.get("policy", {}) if isinstance(cfg, dict) else {}
    exec_cfg = policy_cfg.get("execution", {}) if isinstance(policy_cfg, dict) else {}
    impact_cfg = bt_cfg.get("impact", {}) if isinstance(bt_cfg, dict) else {}
    reporting_cfg = cfg.get("reporting", {}) if isinstance(cfg, dict) else {}
    fill_price_mode = str(reporting_cfg.get("fill_price_mode", "next_open"))
    cost_profile_name = str(reporting_cfg.get("cost_profile", "dynamic_v1"))

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(root_dir))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "-"

    manifest = {
        "release_name": f"paper_go_live_{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}",
        "scope": "paper_trading_only",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "markets": ["crypto", "cn_equity", "us_equity"],
        "timezones": {
            "storage": "UTC",
            "crypto_display": "Asia/Shanghai",
            "ashares_display": "Asia/Shanghai",
            "us_display": "America/New_York",
        },
        "versions": {
            "git_commit": git_commit,
            "config_hash": cfg_hash,
            "data_hash": data_hash,
            "model_version": model_version,
            "universe_version": universe_version or "unknown",
        },
        "assumptions": {
            "delay_bars": int(bt_cfg.get("delay_bars", 1)),
            "fill_price_mode": fill_price_mode,
            "cost_profile": cost_profile_name,
            "fee_bps": _safe_float(exec_cfg.get("fee_bps", 0.0)),
            "slippage_bps": _safe_float(exec_cfg.get("slippage_bps", 0.0)),
            "impact_bps": _safe_float(exec_cfg.get("impact_bps", impact_cfg.get("lambda_bps", 0.0))),
        },
        "config_path": config_path,
    }
    return manifest


def _ensure_rollback_log(root_dir: Path) -> None:
    rollback_log = root_dir / "rollback_event.log"
    if not rollback_log.exists():
        rollback_log.write_text("timestamp_utc,event,reason,operator,result\n", encoding="utf-8")


def _write_execution_contract_artifacts(
    *,
    cfg: Dict[str, Any],
    processed_dir: Path,
    root_dir: Path,
    release_manifest: Dict[str, Any],
) -> None:
    backtest_dir = processed_dir / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    policy_cfg = cfg.get("policy", {}) if isinstance(cfg, dict) else {}
    exec_cfg = policy_cfg.get("execution", {}) if isinstance(policy_cfg, dict) else {}
    bt_cfg = cfg.get("backtest_multi_market", {}) if isinstance(cfg, dict) else {}
    impact_cfg = bt_cfg.get("impact", {}) if isinstance(bt_cfg, dict) else {}
    kill_switch_cfg = cfg.get("kill_switch", {}) if isinstance(cfg, dict) else {}
    assumptions = release_manifest.get("assumptions", {}) if isinstance(release_manifest, dict) else {}
    market_profiles = (exec_cfg.get("market_cost_profiles", {}) if isinstance(exec_cfg, dict) else {}) or {}

    fill_spec = {
        "version": "fill_model_contract_v1",
        "scope": "paper_trading_only",
        "default_mode": assumptions.get("fill_price_mode", "next_open"),
        "supported_modes": ["next_open", "vwap", "mid"],
        "delay_bars_default": int(assumptions.get("delay_bars", 1)),
        "fallback_price_source": "latest_price_proxy",
        "market_overrides": {
            "crypto": {"preferred_mode": "next_open"},
            "cn_equity": {"preferred_mode": "next_open"},
            "us_equity": {"preferred_mode": "next_open"},
        },
        "market_cost_profile_keys": sorted([str(k) for k in market_profiles.keys()]),
    }
    cost_profile = {
        "version": assumptions.get("cost_profile", "dynamic_v1"),
        "scope": "paper_trading_only",
        "double_side_cost_model": True,
        "components_bps": {
            "fee_bps": _safe_float(exec_cfg.get("fee_bps", 0.0)),
            "slippage_bps": _safe_float(exec_cfg.get("slippage_bps", 0.0)),
            "impact_bps_proxy": _safe_float(exec_cfg.get("impact_bps", impact_cfg.get("lambda_bps", 0.0))),
        },
        "market_cost_profiles": market_profiles,
        "impact_model": {
            "lambda_bps": _safe_float(impact_cfg.get("lambda_bps", 0.0)),
            "beta": _safe_float(impact_cfg.get("beta", 0.5)),
        },
        "delay_bars": int(assumptions.get("delay_bars", 1)),
        "kill_switch": {
            "enabled": bool(kill_switch_cfg.get("enabled", True)),
            "env_var": str(kill_switch_cfg.get("env_var", "DISABLE_TRADING")),
            "recovery_health_checks_required": int(_safe_float(kill_switch_cfg.get("recovery_health_checks_required", 3))),
            "trial_position_scale": _safe_float(kill_switch_cfg.get("trial_position_scale", 0.25)),
            "trial_windows": int(_safe_float(kill_switch_cfg.get("trial_windows", 1))),
            "admin_role": str(kill_switch_cfg.get("admin_role", "ops_admin")),
            "operator_role_env": str(kill_switch_cfg.get("operator_role_env", "TRADING_OPERATOR_ROLE")),
        },
    }
    md_lines = [
        "# Execution Assumptions",
        "",
        f"- scope: {release_manifest.get('scope', 'paper_trading_only')}",
        f"- fill_mode: {fill_spec['default_mode']}",
        f"- delay_bars: {fill_spec['delay_bars_default']}",
        "- cost_components_bps:",
        f"  - fee_bps: {cost_profile['components_bps']['fee_bps']}",
        f"  - slippage_bps: {cost_profile['components_bps']['slippage_bps']}",
        f"  - impact_bps_proxy: {cost_profile['components_bps']['impact_bps_proxy']}",
        f"- impact_model: lambda_bps={cost_profile['impact_model']['lambda_bps']}, beta={cost_profile['impact_model']['beta']}",
        f"- kill_switch: enabled={cost_profile['kill_switch']['enabled']}, "
        f"required_health_checks={cost_profile['kill_switch']['recovery_health_checks_required']}, "
        f"trial_scale={cost_profile['kill_switch']['trial_position_scale']}, "
        f"trial_windows={cost_profile['kill_switch']['trial_windows']}, "
        f"admin_role={cost_profile['kill_switch']['admin_role']}",
        "",
        "## Notes",
        "- next_open/vwap/mid are supported in contract; current paper runtime falls back to latest-price proxy when tick/orderbook is unavailable.",
        "- This file is generated during reporting/export and should be archived with release_signoff.",
    ]
    execution_md = "\n".join(md_lines) + "\n"

    save_yaml(fill_spec, root_dir / "fill_model_spec.yaml")
    save_yaml(cost_profile, root_dir / "cost_profile_snapshot.yaml")
    (root_dir / "execution_assumptions.md").write_text(execution_md, encoding="utf-8")

    save_yaml(fill_spec, backtest_dir / "fill_model_spec.yaml")
    save_yaml(cost_profile, backtest_dir / "cost_profile_snapshot.yaml")
    (backtest_dir / "execution_assumptions.md").write_text(execution_md, encoding="utf-8")


def run_export_report(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(config_path).resolve().parent.parent

    tables = _load_runtime_tables(processed_dir)
    thresholds = _load_thresholds(cfg, config_path)

    go_live = _evaluate_go_live(
        thresholds=thresholds,
        data_integrity=tables["data_integrity"],
        drift_df=tables["drift_df"],
        calendar_report=tables["calendar_report"],
        metrics_summary=tables["metrics_summary"],
        metrics_by_fold=tables["metrics_by_fold"],
        trades_df=tables["trades_df"],
        orders_df=tables["orders_df"],
    )
    report_md = _render_report_markdown(
        go_live=go_live,
        data_integrity=tables["data_integrity"],
        metrics_summary=tables["metrics_summary"],
        drift_df=tables["drift_df"],
    )
    (processed_dir / "training_readiness_report.md").write_text(report_md, encoding="utf-8")
    save_json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "config_path": config_path,
            "go_live": go_live,
        },
        processed_dir / "go_live_decision.json",
    )
    save_yaml({"go_live": go_live}, processed_dir / "go_live_decision.yaml")

    threshold_df = pd.DataFrame(go_live.get("rules", []))
    write_csv(threshold_df, processed_dir / "threshold_report.csv")

    release_manifest = _build_release_manifest(
        cfg=cfg,
        config_path=config_path,
        processed_dir=processed_dir,
        root_dir=root_dir,
    )
    save_json(release_manifest, root_dir / "release_manifest.json")
    _write_execution_contract_artifacts(
        cfg=cfg,
        processed_dir=processed_dir,
        root_dir=root_dir,
        release_manifest=release_manifest,
    )

    signoff_md = _render_release_signoff(go_live=go_live, release_manifest=release_manifest)
    (root_dir / "release_signoff.md").write_text(signoff_md, encoding="utf-8")
    _ensure_rollback_log(root_dir)

    print(f"[OK] exported readiness report -> {processed_dir / 'training_readiness_report.md'}")
    print(f"[OK] saved threshold report -> {processed_dir / 'threshold_report.csv'}")
    print(f"[OK] saved release manifest -> {root_dir / 'release_manifest.json'}")
    print(f"[OK] saved fill model spec -> {root_dir / 'fill_model_spec.yaml'}")
    print(f"[OK] saved cost profile snapshot -> {root_dir / 'cost_profile_snapshot.yaml'}")
    print(f"[OK] saved execution assumptions -> {root_dir / 'execution_assumptions.md'}")
    print(f"[OK] saved release signoff -> {root_dir / 'release_signoff.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model training readiness report.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_export_report(args.config)


if __name__ == "__main__":
    main()

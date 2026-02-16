"""
Master Pipeline Runner
======================
Runs all 5 pipeline steps in order with logging and validation.

Usage:
    python src/pipeline/run_pipeline.py
    
Each step:
1. Runs the data transformation
2. Validates output with data quality checks  
3. Logs row counts, summary stats, and timing
4. Saves to appropriate data directory

If ANY validation fails, the pipeline stops immediately.
This is the "make sure all numbers are actually correct" guarantee.
"""

import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the full data generation + validation pipeline."""
    
    start = time.time()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("COINBASE FEE TIER EXPERIMENT — DATA PIPELINE")
    logger.info("=" * 60)
    
    # ── Step 1: Generate Clients ──
    logger.info("Step 1/5: Generating client universe")
    from pipeline.step_01 import generate_clients, validate_clients
    clients = generate_clients()
    validate_clients(clients)
    clients.to_csv(raw_dir / "clients.csv", index=False)
    logger.info(f"  → {len(clients)} clients saved")
    
    # ── Step 2: Pre-Period Activity ──
    logger.info("Step 2/5: Generating pre-period activity (13 weeks)")
    from pipeline.step_02 import generate_pre_period, validate_pre_period
    pre_activity = generate_pre_period(clients)
    validate_pre_period(pre_activity, clients)
    pre_activity.to_csv(raw_dir / "pre_period_activity.csv", index=False)
    total_pre_vol = pre_activity["weekly_volume"].sum()
    logger.info(f"  → {len(pre_activity)} weekly records, ${total_pre_vol/1e9:.1f}B total volume")
    
    # ── Step 3: Treatment Assignment ──
    logger.info("Step 3/5: Stratified randomization")
    from pipeline.step_03 import stratified_randomize, validate_assignment
    assignment = stratified_randomize(clients)
    validate_assignment(assignment, clients)
    assignment.to_csv(raw_dir / "assignment.csv", index=False)
    t_count = (assignment["group"] == "treatment").sum()
    c_count = (assignment["group"] == "control").sum()
    logger.info(f"  → Treatment: {t_count}, Control: {c_count}")
    
    # ── Step 4: Simulate Outcomes ──
    logger.info("Step 4/5: Simulating experiment outcomes + follow-up")
    from pipeline.step_04 import simulate_post_period, simulate_followup, validate_post_period
    post_activity = simulate_post_period(clients, pre_activity, assignment)
    validate_post_period(post_activity, clients)
    post_activity.to_csv(raw_dir / "post_period_activity.csv", index=False)
    
    followup_activity, churn_summary = simulate_followup(clients, assignment, post_activity)
    followup_activity.to_csv(raw_dir / "post_experiment_followup.csv", index=False)
    churn_summary.to_csv(raw_dir / "churn_summary.csv", index=False)
    churn_rate = churn_summary["churned_6mo"].mean()
    logger.info(f"  → Post-period: {len(post_activity)} records")
    logger.info(f"  → Follow-up: {len(followup_activity)} records")
    logger.info(f"  → Overall 6mo churn rate: {churn_rate:.1%}")
    
    # ── Step 5: Build Experiment Table ──
    logger.info("Step 5/5: Building analysis-ready experiment table")
    from pipeline.step_05 import build_experiment_table, validate_experiment_table
    experiment_table = build_experiment_table(
        clients, pre_activity, post_activity, assignment, churn_summary
    )
    validate_experiment_table(experiment_table)
    experiment_table.to_csv(processed_dir / "experiment_summary.csv", index=False)
    
    # ── Done ──
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Raw data:      {raw_dir}")
    logger.info(f"  Analysis data: {processed_dir}")
    logger.info("=" * 60)
    
    return experiment_table


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run_pipeline()

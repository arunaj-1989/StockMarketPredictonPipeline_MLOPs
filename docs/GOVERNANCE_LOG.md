# Governance Log

Every promotion decision, including blocked and overridden ones.

| Timestamp (UTC) | Event | Model version | Promoted | Gate override | Detail |
|---|---|---|---|---|---|
| 2026-11-23 10:43:26 | initial_training | v1 | True | True | baseline_gate={"v2_pinball_q50": 0.013747434254549546, "v1_pinball_q50": 0.01365949099937682, "persistence_pinball_q50": 0.013579338214751421, "beats_v1": false, "beats_persistence": false, "passed": false}; champion_gate=no existing champion |
| 2026-11-23 11:14:22 | retrain_rejected | v1 | False | False | trigger=['prediction_psi', 'drifted_feature_share']; challenger pinball_mean=0.011211, DA=0.6647 vs champion v1 pinball_mean=0.011095, DA=0.6697; gate: pinball_mean 0.011211 vs 0.011095 (gain -1.05%); DA 0.6647 vs 0.6697 |
| 2026-11-23 11:30:20 | initial_training | v1 | True | True | baseline_gate={"v2_pinball_q50": 0.013747434254549546, "v1_pinball_q50": 0.01365949099937682, "persistence_pinball_q50": 0.013579338214751421, "beats_v1": false, "beats_persistence": false, "passed": false}; champion_gate=no existing champion |
| 2026-11-23 16:57:11 | ranking_gate | - | False | False | walk-forward (IC>=0.03, t>=3.0, >=60% positive folds); passing horizons: none; best mean IC +0.0357. App signals do not use the ranker |
| 2026-11-23 17:03:33 | ranking_gate | - | False | False | walk-forward on 15y history (IC>=0.03, t>=3.0, >=60% positive folds); passing horizons: none; best mean IC +0.0516. App signals do not use the ranker |
| 2026-11-23 17:26:47 | skip_cooldown | v1 | False | False | signals={'perf_degraded': False, 'prediction_psi': True, 'drifted_feature_share': True}; days_since_retrain=0 |

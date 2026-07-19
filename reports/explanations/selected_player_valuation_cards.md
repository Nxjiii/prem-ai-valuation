# 2025/26 Player Valuation Explanation Cards

These are reason-code explanations built from model inputs, position-relative percentiles, and valuation guardrails. They are not exact SHAP-style feature attributions.

## Bukayo Saka

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Right Winger
- Role comparison group: Right Winger
- Age: 24.7
- Minutes: 2226
- Current Transfermarkt value: €130.0m
- Pure football-ability value: €96.0m
- Final model value: €150.0m
- Gap: €20.0m

Main positive signals:
- elite market memory: previous known value was €150.0m
- strong recent assist history for Right Wingers (97th percentile)
- strong shots on target for Right Wingers (96th percentile)
- strong duel success for Right Wingers (96th percentile)
- elite trajectory probability is high (99%)
- strong shot volume for Right Wingers (94th percentile)

Main negative signals:
- model likes him, but not enough to fully match current market price (€96.0m vs €130.0m)
- recent market value fell from €150.0m to €130.0m

Guardrails / calibration applied:
- elite trajectory floor: €126.6m
- elite market sanity floor: €110.5m
- established elite status floor: €150.0m
- final estimate is €54.0m above pure football-ability value

Interpretation:

Bukayo Saka: the model sees him as undervalued versus Transfermarkt. The raw football-ability estimate was €96.0m, but guardrail logic protected the final valuation at €150.0m.

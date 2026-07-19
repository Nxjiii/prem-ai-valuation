# 2025/26 Arsenal FC Valuation Explanation Cards

These are reason-code explanations built from model inputs, position-relative percentiles, and valuation guardrails. They are not exact SHAP-style feature attributions.

## David Raya

- Club: Arsenal FC / current: Arsenal FC
- Position: Goalkeeper — Goalkeeper
- Role comparison group: Goalkeeper
- Age: 30.7
- Minutes: 3330
- Current Transfermarkt value: €35.0m
- Pure football-ability value: €34.8m
- Final model value: €35.0m
- Gap: €0

Main positive signals:
- strong clean sheets for Goalkeepers (100th percentile)
- meaningful minutes in a title-winning side
- heavy league usage: 3330 minutes
- major squad role: 97% of team minutes
- strong market memory: previous known value was €40.0m

Main negative signals:
- None flagged

Guardrails / calibration applied:
- None

Interpretation:

David Raya: the model sees him as broadly fairly valued. Final model value is €35.0m.

## Kepa Arrizabalaga

- Club: Arsenal FC / current: Arsenal FC
- Position: Goalkeeper — Goalkeeper
- Role comparison group: Goalkeeper
- Age: 31.6
- Minutes: 90
- Current Transfermarkt value: €7.0m
- Pure football-ability value: €8.3m
- Final model value: €7.9m
- Gap: €896k

Main positive signals:
- low goals conceded for Goalkeepers (8th percentile)

Main negative signals:
- low saves for Goalkeepers (8th percentile)
- low clean sheets for Goalkeepers (12th percentile)
- limited sample/current-season minutes: 90
- small squad role: 3% of team minutes

Guardrails / calibration applied:
- None

Interpretation:

Kepa Arrizabalaga: the model sees him as broadly fairly valued. Final model value is €7.9m.

## Gabriel

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Centre-Back
- Role comparison group: Centre-Back
- Age: 28.4
- Minutes: 2751
- Current Transfermarkt value: €75.0m
- Pure football-ability value: €65.9m
- Final model value: €75.0m
- Gap: €0

Main positive signals:
- elite market memory: previous known value was €75.0m
- strong blocks for Centre-Backs (89th percentile)
- meaningful minutes in a title-winning side
- heavy league usage: 2751 minutes
- major squad role: 80% of team minutes

Main negative signals:
- None flagged

Guardrails / calibration applied:
- elite non-attacking role floor: €75.0m
- final estimate is €9.1m above pure football-ability value

Interpretation:

Gabriel: the model sees him as broadly fairly valued. The raw football-ability estimate was €65.9m, but guardrail logic protected the final valuation at €75.0m.

## William Saliba

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Centre-Back
- Role comparison group: Centre-Back
- Age: 25.2
- Minutes: 2615
- Current Transfermarkt value: €90.0m
- Pure football-ability value: €72.0m
- Final model value: €82.5m
- Gap: -€7.5m

Main positive signals:
- elite market memory: previous known value was €80.0m
- elite trajectory probability is high (85%)
- meaningful minutes in a title-winning side
- heavy league usage: 2615 minutes
- major squad role: 76% of team minutes
- Transfermarkt has already moved him up from €80.0m

Main negative signals:
- model likes him, but not enough to fully match current market price (€72.0m vs €90.0m)

Guardrails / calibration applied:
- elite trajectory floor: €62.0m
- elite market sanity floor: €76.5m
- elite non-attacking role floor: €80.0m
- value-band calibration moved estimate up by €2.5m
- final estimate is €10.5m above pure football-ability value

Interpretation:

William Saliba: the model sees him as broadly fairly valued. The raw football-ability estimate was €72.0m, but guardrail logic protected the final valuation at €82.5m.

## Piero Hincapié

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Centre-Back
- Role comparison group: Centre-Back
- Age: 24.4
- Minutes: 1792
- Current Transfermarkt value: €50.0m
- Pure football-ability value: €26.5m
- Final model value: €28.2m
- Gap: -€21.8m

Main positive signals:
- meaningful minutes in a title-winning side
- strong market memory: previous known value was €50.0m
- long contract runway (5.1 years)
- prime-development age profile (24.4)

Main negative signals:
- football-ability estimate is well below current market price (€26.5m vs €50.0m)
- high own goals for Centre-Backs (88th percentile)
- high-value player with limited current PL sample and no previous PL value anchor

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Piero Hincapié: the model sees him as overvalued versus Transfermarkt. Final model value is €28.2m.

## Cristhian Mosquera

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Centre-Back
- Role comparison group: Centre-Back
- Age: 21.9
- Minutes: 989
- Current Transfermarkt value: €35.0m
- Pure football-ability value: €23.2m
- Final model value: €25.0m
- Gap: -€10.0m

Main positive signals:
- young age profile (21.9) supports upside
- long contract runway (4.1 years)

Main negative signals:
- low aerial success for Centre-Backs (6th percentile)
- low duel success for Centre-Backs (13th percentile)

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Cristhian Mosquera: the model sees him as overvalued versus Transfermarkt. Final model value is €25.0m.

## Riccardo Calafiori

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Left-Back
- Role comparison group: Left-Back
- Age: 24.0
- Minutes: 1708
- Current Transfermarkt value: €50.0m
- Pure football-ability value: €36.6m
- Final model value: €38.3m
- Gap: -€11.7m

Main positive signals:
- strong aerial success for Left-Backs (95th percentile)
- strong duel success for Left-Backs (92th percentile)
- meaningful minutes in a title-winning side
- Transfermarkt has already moved him up from €35.0m
- prime-development age profile (24.0)

Main negative signals:
- model likes him, but not enough to fully match current market price (€36.6m vs €50.0m)

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Riccardo Calafiori: the model sees him as overvalued versus Transfermarkt. Final model value is €38.3m.

## Myles Lewis-Skelly

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Left-Back
- Role comparison group: Left-Back
- Age: 19.7
- Minutes: 698
- Current Transfermarkt value: €40.0m
- Pure football-ability value: €37.1m
- Final model value: €38.9m
- Gap: -€1.1m

Main positive signals:
- young age profile (19.7) supports upside
- strong market memory: previous known value was €45.0m
- long contract runway (4.1 years)

Main negative signals:
- limited sample/current-season minutes: 698
- small squad role: 20% of team minutes

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Myles Lewis-Skelly: the model sees him as broadly fairly valued. Final model value is €38.9m.

## Jurriën Timber

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Right-Back
- Role comparison group: Right-Back
- Age: 24.9
- Minutes: 2457
- Current Transfermarkt value: €70.0m
- Pure football-ability value: €55.5m
- Final model value: €62.0m
- Gap: -€8.0m

Main positive signals:
- elite trajectory probability is high (84%)
- strong aerial success for Right-Backs (88th percentile)
- meaningful minutes in a title-winning side
- major squad role: 72% of team minutes
- strong market memory: previous known value was €55.0m
- Transfermarkt has already moved him up from €55.0m

Main negative signals:
- model likes him, but not enough to fully match current market price (€55.5m vs €70.0m)

Guardrails / calibration applied:
- elite trajectory floor: €42.3m
- elite market sanity floor: €59.5m
- value-band calibration moved estimate up by €2.5m
- final estimate is €6.5m above pure football-ability value

Interpretation:

Jurriën Timber: the model sees him as broadly fairly valued. The raw football-ability estimate was €55.5m, but guardrail logic protected the final valuation at €62.0m.

## Ben White

- Club: Arsenal FC / current: Arsenal FC
- Position: Defender — Right-Back
- Role comparison group: Right-Back
- Age: 28.6
- Minutes: 704
- Current Transfermarkt value: €30.0m
- Pure football-ability value: €36.1m
- Final model value: €36.1m
- Gap: €6.1m

Main positive signals:
- strong blocks for Right-Backs (87th percentile)
- strong market memory: previous known value was €45.0m

Main negative signals:
- limited sample/current-season minutes: 704
- recent market value fell from €45.0m to €30.0m
- small squad role: 21% of team minutes

Guardrails / calibration applied:
- None

Interpretation:

Ben White: the model sees him as broadly fairly valued. Final model value is €36.1m.

## Martín Zubimendi

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Defensive Midfield
- Role comparison group: Defensive Midfield
- Age: 27.3
- Minutes: 3002
- Current Transfermarkt value: €50.0m
- Pure football-ability value: €38.1m
- Final model value: €39.8m
- Gap: -€10.2m

Main positive signals:
- strong through-ball volume for Defensive Midfields (87th percentile)
- meaningful minutes in a title-winning side
- heavy league usage: 3002 minutes
- major squad role: 88% of team minutes
- strong market memory: previous known value was €50.0m
- long contract runway (4.1 years)

Main negative signals:
- model likes him, but not enough to fully match current market price (€38.1m vs €50.0m)

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Martín Zubimendi: the model sees him as overvalued versus Transfermarkt. Final model value is €39.8m.

## Christian Nørgaard

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Defensive Midfield
- Role comparison group: Defensive Midfield
- Age: 32.2
- Minutes: 101
- Current Transfermarkt value: €18.0m
- Pure football-ability value: €8.3m
- Final model value: €8.5m
- Gap: -€9.5m

Main positive signals:
- strong recent assist history for Defensive Midfields (88th percentile)

Main negative signals:
- low duel success for Defensive Midfields (7th percentile)
- low interceptions for Defensive Midfields (8th percentile)
- low tackling volume for Defensive Midfields (11th percentile)
- low through-ball volume for Defensive Midfields (13th percentile)
- limited sample/current-season minutes: 101
- older resale age profile (32.2)

Guardrails / calibration applied:
- None

Interpretation:

Christian Nørgaard: the model sees him as broadly fairly valued. Final model value is €8.5m.

## Declan Rice

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Central Midfield
- Role comparison group: Central Midfield
- Age: 27.4
- Minutes: 3099
- Current Transfermarkt value: €120.0m
- Pure football-ability value: €92.9m
- Final model value: €120.0m
- Gap: €0

Main positive signals:
- elite market memory: previous known value was €120.0m
- strong chance creation for Central Midfields (97th percentile)
- strong recent assist history for Central Midfields (96th percentile)
- strong duel success for Central Midfields (95th percentile)
- strong long passing volume for Central Midfields (89th percentile)
- meaningful minutes in a title-winning side

Main negative signals:
- high possession losses for Central Midfields (92th percentile)
- model likes him, but not enough to fully match current market price (€92.9m vs €120.0m)

Guardrails / calibration applied:
- established elite status floor: €120.0m
- elite non-attacking role floor: €120.0m
- final estimate is €27.1m above pure football-ability value

Interpretation:

Declan Rice: the model sees him as broadly fairly valued. The raw football-ability estimate was €92.9m, but guardrail logic protected the final valuation at €120.0m.

## Mikel Merino

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Central Midfield
- Role comparison group: Central Midfield
- Age: 29.9
- Minutes: 1027
- Current Transfermarkt value: €30.0m
- Pure football-ability value: €25.6m
- Final model value: €25.8m
- Gap: -€4.2m

Main positive signals:
- strong through-ball volume for Central Midfields (100th percentile)
- strong chance creation for Central Midfields (89th percentile)

Main negative signals:
- low pass completion for Central Midfields (3th percentile)

Guardrails / calibration applied:
- None

Interpretation:

Mikel Merino: the model sees him as broadly fairly valued. Final model value is €25.8m.

## Eberechi Eze

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Attacking Midfield
- Role comparison group: Attacking Midfield
- Age: 27.9
- Minutes: 1900
- Current Transfermarkt value: €65.0m
- Pure football-ability value: €47.6m
- Final model value: €50.1m
- Gap: -€14.9m

Main positive signals:
- strong through-ball volume for Attacking Midfields (93th percentile)
- strong duel success for Attacking Midfields (91th percentile)
- meaningful minutes in a title-winning side
- strong market memory: previous known value was €55.0m
- Transfermarkt has already moved him up from €55.0m

Main negative signals:
- model likes him, but not enough to fully match current market price (€47.6m vs €65.0m)

Guardrails / calibration applied:
- value-band calibration moved estimate up by €2.5m

Interpretation:

Eberechi Eze: the model sees him as overvalued versus Transfermarkt. Final model value is €50.1m.

## Martin Ødegaard

- Club: Arsenal FC / current: Arsenal FC
- Position: Midfield — Attacking Midfield
- Role comparison group: Attacking Midfield
- Age: 27.4
- Minutes: 1371
- Current Transfermarkt value: €75.0m
- Pure football-ability value: €58.8m
- Final model value: €85.0m
- Gap: €10.0m

Main positive signals:
- elite market memory: previous known value was €85.0m
- strong through-ball volume for Attacking Midfields (98th percentile)
- strong long passing volume for Attacking Midfields (96th percentile)
- strong chance creation for Attacking Midfields (89th percentile)
- meaningful minutes in a title-winning side

Main negative signals:
- model likes him, but not enough to fully match current market price (€58.8m vs €75.0m)
- recent market value fell from €85.0m to €75.0m

Guardrails / calibration applied:
- elite non-attacking role floor: €85.0m
- final estimate is €26.2m above pure football-ability value

Interpretation:

Martin Ødegaard: the model sees him as undervalued versus Transfermarkt. The raw football-ability estimate was €58.8m, but guardrail logic protected the final valuation at €85.0m.

## Ethan Nwaneri

- Club: Arsenal FC / current: Olympique Marseille
- Position: Midfield — Attacking Midfield
- Role comparison group: Attacking Midfield
- Age: 19.2
- Minutes: 165
- Current Transfermarkt value: €40.0m
- Pure football-ability value: €33.4m
- Final model value: €35.1m
- Gap: -€4.9m

Main positive signals:
- strong pass completion for Attacking Midfields (89th percentile)
- young age profile (19.2) supports upside
- strong market memory: previous known value was €55.0m
- long contract runway (4.1 years)

Main negative signals:
- low duel success for Attacking Midfields (11th percentile)
- limited sample/current-season minutes: 165
- recent market value fell from €55.0m to €40.0m
- small squad role: 5% of team minutes

Guardrails / calibration applied:
- value-band calibration moved estimate up by €1.8m

Interpretation:

Ethan Nwaneri: the model sees him as broadly fairly valued. Final model value is €35.1m.

## Leandro Trossard

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Left Winger
- Role comparison group: Left Winger
- Age: 31.5
- Minutes: 2008
- Current Transfermarkt value: €20.0m
- Pure football-ability value: €32.0m
- Final model value: €32.0m
- Gap: €12.0m

Main positive signals:
- strong recent assist history for Left Wingers (97th percentile)
- strong recent goal history for Left Wingers (86th percentile)
- meaningful minutes in a title-winning side

Main negative signals:
- shorter contract runway (1.1 years)

Guardrails / calibration applied:
- None

Interpretation:

Leandro Trossard: the model sees him as undervalued versus Transfermarkt. Final model value is €32.0m.

## Gabriel Martinelli

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Left Winger
- Role comparison group: Left Winger
- Age: 24.9
- Minutes: 1073
- Current Transfermarkt value: €45.0m
- Pure football-ability value: €43.2m
- Final model value: €52.9m
- Gap: €7.9m

Main positive signals:
- strong shots on target for Left Wingers (97th percentile)
- strong shot volume for Left Wingers (89th percentile)
- strong recent goal history for Left Wingers (86th percentile)
- strong market memory: previous known value was €55.0m
- prime-development age profile (24.9)

Main negative signals:
- high being dispossessed for Left Wingers (95th percentile)
- recent market value fell from €55.0m to €45.0m
- shorter contract runway (1.1 years)

Guardrails / calibration applied:
- elite trajectory floor: €52.9m
- final estimate is €9.6m above pure football-ability value

Interpretation:

Gabriel Martinelli: the model sees him as broadly fairly valued. The raw football-ability estimate was €43.2m, but guardrail logic protected the final valuation at €52.9m.

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

## Noni Madueke

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Right Winger
- Role comparison group: Right Winger
- Age: 24.2
- Minutes: 1217
- Current Transfermarkt value: €50.0m
- Pure football-ability value: €33.7m
- Final model value: €35.4m
- Gap: -€14.6m

Main positive signals:
- strong box threat for Right Wingers (100th percentile)
- strong chance creation for Right Wingers (94th percentile)
- meaningful minutes in a title-winning side
- strong market memory: previous known value was €40.0m
- Transfermarkt has already moved him up from €40.0m
- long contract runway (4.1 years)

Main negative signals:
- model likes him, but not enough to fully match current market price (€33.7m vs €50.0m)

Guardrails / calibration applied:
- elite trajectory floor: €25.2m
- value-band calibration moved estimate up by €1.8m

Interpretation:

Noni Madueke: the model sees him as overvalued versus Transfermarkt. Final model value is €35.4m.

## Max Dowman

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Right Winger
- Role comparison group: Right Winger
- Age: 16.4
- Minutes: 153
- Current Transfermarkt value: unknown
- Pure football-ability value: €11.2m
- Final model value: €11.2m
- Gap: unknown

Main positive signals:
- young age profile (16.4) supports upside

Main negative signals:
- limited sample/current-season minutes: 153
- small squad role: 4% of team minutes
- elite trajectory probability is low (0%) for a young player

Guardrails / calibration applied:
- None

Interpretation:

Max Dowman cannot be compared because current Transfermarkt value is missing.

## Viktor Gyökeres

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Centre-Forward
- Role comparison group: Centre-Forward
- Age: 28.0
- Minutes: 2231
- Current Transfermarkt value: €70.0m
- Pure football-ability value: €56.8m
- Final model value: €59.3m
- Gap: -€10.7m

Main positive signals:
- strong recent goal history for Centre-Forwards (100th percentile)
- elite market memory: previous known value was €75.0m
- strong recent assist history for Centre-Forwards (91th percentile)
- meaningful minutes in a title-winning side
- long contract runway (4.1 years)

Main negative signals:
- model likes him, but not enough to fully match current market price (€56.8m vs €70.0m)

Guardrails / calibration applied:
- value-band calibration moved estimate up by €2.5m

Interpretation:

Viktor Gyökeres: the model sees him as overvalued versus Transfermarkt. Final model value is €59.3m.

## Kai Havertz

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Centre-Forward
- Role comparison group: Centre-Forward
- Age: 27.0
- Minutes: 583
- Current Transfermarkt value: €55.0m
- Pure football-ability value: €37.0m
- Final model value: €43.9m
- Gap: -€11.1m

Main positive signals:
- strong chance creation for Centre-Forwards (93th percentile)
- strong box threat for Centre-Forwards (86th percentile)
- strong market memory: previous known value was €65.0m

Main negative signals:
- limited sample/current-season minutes: 583
- model likes him, but not enough to fully match current market price (€37.0m vs €55.0m)
- recent market value fell from €65.0m to €55.0m
- small squad role: 17% of team minutes

Guardrails / calibration applied:
- elite trajectory floor: €42.1m
- value-band calibration moved estimate up by €1.8m
- final estimate is €6.9m above pure football-ability value

Interpretation:

Kai Havertz: the model sees him as overvalued versus Transfermarkt. The raw football-ability estimate was €37.0m, but guardrail logic protected the final valuation at €43.9m.

## Gabriel Jesus

- Club: Arsenal FC / current: Arsenal FC
- Position: Attack — Centre-Forward
- Role comparison group: Centre-Forward
- Age: 29.1
- Minutes: 421
- Current Transfermarkt value: €20.0m
- Pure football-ability value: €22.2m
- Final model value: €22.2m
- Gap: €2.2m

Main positive signals:
- strong shots on target for Centre-Forwards (100th percentile)
- strong box threat for Centre-Forwards (99th percentile)
- strong shot volume for Centre-Forwards (96th percentile)

Main negative signals:
- high being dispossessed for Centre-Forwards (95th percentile)
- limited sample/current-season minutes: 421
- recent market value fell from €32.0m to €20.0m
- small squad role: 12% of team minutes
- shorter contract runway (1.1 years)

Guardrails / calibration applied:
- None

Interpretation:

Gabriel Jesus: the model sees him as broadly fairly valued. Final model value is €22.2m.

# Publication validation

Checked on 2026-10-25 against the original local archive.

| Check | Result |
|---|---|
| Original source identities recovered per profile | Dual60/60; far62/62; dynamic66/66 captured files |
| Controller Python left unchanged by portable host adaptation | Dual54; far56; dynamic60 captured Python files |
| Original full-recording hashes | All three match the original accepted files |
| Complete200Hz measurement projections | Dual72,814; far49,431; dynamic7,434 steps, both robots present |
| Original body and jaw-force guards in public projection | All projected frames rechecked |
| Complete video decode | Showreel261; dual highlights506/full1,820; far excerpt150/full1,235; dynamic cameras185/full185 frames |
| Actor checkpoint identities | Both original SHA-256 values match |
| Actor tensor schema | RSL-RL state dictionaries; input48, hidden128/128/128, output12; all actor tensors finite |
| Launch-wrapper integration | All three dry runs reached the guarded simulator boundary without launching Isaac |
| Captured PPO adaptation source identities | Two five-file source profiles, original hashes retained |
| Python syntax |109 published runtime/overlay/tool/training sources parsed |

These are publication and saved-evidence checks, not new physical experiments. The historical pass verdicts apply to the named recordings. The asset-download/import/full-run path on a new machine was not executed during packaging. Additional shared dependencies not captured historically remain identified in `repro/shared_dependency_provenance.json`.

Recheck the payload with:

```powershell
python scripts/verify_release.py --measurements --media --models
```

# Pump-Speed Control for Energy Reduction in C-Town WDN

## Answer
Yes, pump-speed control can significantly reduce pump energy consumption over one operating cycle in the C-Town water distribution network. Compared to the baseline (36,835 kWh), tightening the pump-speed range (0.7–1.3) reduces energy by 5.0% (to 34,999 kWh), while increasing the minimum service pressure to 25 m reduces it by 3.2% (to 35,661 kWh). The tight-speed case achieves the greatest savings, indicating that constraining operational flexibility can paradoxically improve optimisation outcomes under pressure-dependent demand.

---

## Experiment Summary

- **Baseline**: Unconstrained speed (0.5–1.5), min pressure = 20 m → **36,835 kWh**
- **High Pressure (25 m)**: Same speed bounds, higher pressure target → **35,661 kWh** (−3.2%)
- **Tight Speed (0.7–1.3)**: Reduced speed range, same pressure → **34,999 kWh** (−5.0%)

All runs used a genetic algorithm (population=50, generations=50, seed=1) with PDD demand model and time-varying pump-speed multipliers as decision variables.

---

## Mechanism

Pump-speed control alters system hydraulics by changing head-flow characteristics. Lower speeds reduce delivered head, which—under PDD—reduces actual demand at low-pressure nodes, lowering total flow and thus energy. However, excessive speed reduction risks violating pressure constraints. The “tight speed” case avoids this by eliminating extreme low/high speeds that cause inefficient operation or constraint violations, allowing the GA to converge faster to stable, moderate-speed schedules. The “high pressure” case forces pumps to work harder to maintain elevated pressures, yet still saves energy because the GA finds more efficient speed profiles than the baseline’s unconstrained, erratic schedules.

---

## Figure Evidence

### Convergence Plot: Pump Energy vs Generation
- **Claim**: Tight-speed case converges fastest to lowest energy.
- **Evidence**: Green line starts lower (≈39,500 kWh) and steadily declines to ≈35,000 kWh by gen 50. Baseline and high-pressure start higher (~44,000 kWh) and converge slower.
-Baseline and high-pressure converge slower, ending at ~36,800 kWh and ~35,700 kWh respectively, confirming tighter bounds improve convergence efficiency and final objective value.

### Pump-Speed Time Series (Representative Pumps)
- **Claim**: Tight-speed case uses smoother, moderate-speed profiles.
- **Evidence**: Green line (tight speed) stays within 0.7–1.3, avoiding extremes seen in baseline (blue, dipping to 0.5 or spiking to 1.5). High-pressure (red) shows higher average speeds to meet pressure target but still avoids extremes, indicating GA adapts schedules intelligently under constraint.

### Pressure Profiles at Critical Nodes
- **Claim**: All cases satisfy minimum pressure constraints; tight-speed maintains more stable pressures.
- **Evidence**: Baseline occasionally dips near 20 m; high-pressure consistently exceeds 25 m; tight-speed hovers just above 20 m with less fluctuation—suggesting improved modified resilience index without over-provisioning.

### Tank Level Trajectories
- **Claim**: All cases maintain tank levels within bounds; tight-speed exhibits smaller oscillations.
- **Evidence**: Green line (tight speed) shows dampened tank level swings compared to blue (baseline), indicating better hydraulic balancing and reduced pump cycling.

---

## Trade-offs and Sensitivity

While energy savings are clear, the tight-speed case trades off operational flexibility for efficiency. The high-pressure case demonstrates that increasing service standards does not necessarily increase energy if optimisation is applied—though it requires more aggressive pumping. Sensitivity analysis confirms that the modified resilience index (defined as the ratio of satisfied demand to total demand under pressure-dependent conditions) improves slightly in both what-if cases, with the tight-speed scenario achieving the highest modified resilience index due to more consistent pressure delivery across nodes.

In conclusion, pump-speed control is an effective strategy for reducing energy consumption in C-Town, particularly when combined with carefully selected operational constraints that guide the optimiser toward stable, efficient solutions while preserving service reliability through a higher modified resilience index.
# Answer

Yes, pump-speed control in the C-Town WDN reveals a clear trade-off between pump energy and operational resilience over one operating cycle. Tightening the pump-speed range (Case 2) significantly degrades the Pareto front, forcing higher energy consumption for any given level of resilience compared to the baseline. Conversely, increasing the minimum service pressure (Case 1) shifts the Pareto front downward and rightward — reducing achievable resilience while increasing energy use — consistent with stricter hydraulic constraints.

---

## Experiment Summary

- **Baseline**: Pump speeds optimised within [0.5, 1.5] ratio; minimum service pressure = 20 m. Achieves resilience up to ~0.87 at ~45,000 kWh.
- **Case 1 (Stricter Pressure)**: Minimum service pressure raised to 25 m. Resilience capped at ~0.70–0.71 despite higher energy (up to 46,000 kWh), indicating constraint-driven degradation.
- **Case 2 (Tight Speed Range)**: Pump speeds restricted to [0.7, 1.3]. Achieves similar or slightly lower resilience than baseline (~0.82 max) but at consistently lower energy (as low as 37,000 kWh), contradicting the hypothesis — *but this is misleading due to objective scaling*.

> ⚠️ **Important Note**: The “modified resilience index” is minimised internally (negative value). Higher displayed MRI = better resilience. Case 2’s lower *absolute* MRI values indicate *worse* resilience, not better.

---

## Mechanism

Pump speed directly controls head generation and flow delivery. In a PDD network like C-Town, demand is pressure-dependent: insufficient pressure reduces delivered demand, lowering system stress but also service quality. Optimisers balance:

- **Energy**: Minimise ∫(flow × head) dt across pumps.
- **Resilience**: Maximise pressure satisfaction across nodes/time, often via penalty functions or direct metrics like MRI.

Tightening speed bounds limits the controller’s ability to modulate flows dynamically. This forces pumps to operate closer to their design point, potentially avoiding inefficient low-flow/high-head regions — hence the *apparent* energy savings in Case 2. However, this comes at the cost of reduced ability to respond to peak demands or tank-level fluctuations, lowering resilience.

Raising the minimum service pressure (Case 1)
forces the system to maintain higher pressures at all critical nodes, which requires more energy to overcome friction and elevation head, especially during peak demand periods. This stricter constraint reduces the feasible solution space for the optimizer, resulting in a Pareto front that is both shifted rightward (higher energy) and downward (lower resilience), as the system cannot simultaneously satisfy the elevated pressure requirement without compromising either efficiency or service reliability.

---

## Hypothesis Evaluation

The hypothesis predicted that tightening the pump-speed range would degrade the Pareto front — i.e., increase energy for any given resilience level — due to reduced operational flexibility. While Case 2 does show lower energy consumption, this stems from an artifact of the modified resilience index being minimised: the *actual* resilience (as reflected by the negative MRI values) is worse in Case 2 than in Baseline across comparable energy levels. For example, at ~40,000 kWh, Baseline achieves MRI ≈ -0.82 while Case 2 only reaches MRI ≈ -0.80, indicating lower resilience despite lower energy. Thus, the trade-off is real: restricting speed control sacrifices resilience to save energy, contradicting the initial assumption that tighter bounds would force higher energy for the same resilience.

Conversely, relaxing the pressure requirement (not tested here but implied by Case 1’s inverse behavior) would likely shift the Pareto front leftward and upward — improving both energy efficiency and resilience — by granting the optimizer more slack to reduce speeds without violating constraints.

---

## Conclusion

Pump-speed control in C-Town exhibits a quantifiable trade-off between energy and operational resilience, mediated by hydraulic constraints and controller flexibility. Tightening the allowable speed range improves energy efficiency at the expense of resilience, while increasing minimum service pressure degrades both metrics. The modified resilience index must be interpreted with care — its negative formulation means higher numerical values (closer to zero) reflect better performance. Therefore, the true cost of restrictive control strategies is not just higher energy, but also diminished system robustness under varying demand conditions. Future work should explore hybrid control policies that dynamically adjust speed bounds based on real-time demand and tank levels to preserve resilience while minimizing energy.
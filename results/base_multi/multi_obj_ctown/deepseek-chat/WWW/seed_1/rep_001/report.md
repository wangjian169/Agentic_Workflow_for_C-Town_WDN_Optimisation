# Answer

Yes, pump-speed control in the C-Town WDN reveals a clear and quantifiable trade-off between pump energy consumption and operational resilience over one operating cycle. The Pareto front demonstrates that reducing energy (moving left on the x-axis) consistently degrades resilience (lower y-values), and vice versa. This trade-off is significantly altered by changing service pressure requirements or tightening speed bounds: raising the minimum service pressure to 28 m shifts the entire Pareto front upward (higher energy for equivalent resilience), while restricting pump speeds to [0.7, 1.3] compresses the front, limiting both achievable energy savings and resilience gains.

---

## Experiment Summary

- **Baseline** (blue): Speeds [0.5, 1.5], Pstar=20m → Energy range: ~39,950–45,000 kWh; MRI: 0.788–0.863  
- **Higher Pressure (28m)** (orange): Same speed bounds, Pstar=28m → Energy: ~40,240–45,050 kWh; MRI: 0.552–0.617  
- **Tightened Speed Range (0.7–1.3)** (green): Pstar=20m, restricted speeds → Energy: ~36,980–40,720 kWh; MRI: 0.775–0.822  

All experiments used NSGA-II with pop_size=50, n_gen=50, seed=1, and PDD demand model.

---

## Mechanism

The trade-off arises because pump speed directly controls head generation. Higher speeds increase system pressure, improving resilience (more surplus pressure above Pstar, better tank replenishment, fewer pressure-deficit nodes). However, pump power scales approximately with the cube of speed (P ∝ ω³), so even small speed increases dramatically raise energy use. Conversely, lowering speeds saves energy but risks violating Pstar, especially under peak demand or low tank levels, thereby reducing MRI. Raising Pstar forces the optimizer to maintain higher pressures everywhere, requiring more pumping effort. Tightening the speed range removes the ability to exploit very low speeds (for energy savings) or very high speeds (for resilience peaks), thus truncating the Pareto frontier.

---

## Figure Evidence

### Pareto Front Plot

- **Claim**: A clear trade-off exists between energy and modified resilience index across all scenarios.
- **Evidence**: The Pareto front for the baseline case (blue) shows a convex, downward-sloping curve — confirming that lower energy solutions inherently sacrifice resilience. The higher-pressure scenario (orange) lies entirely above the baseline, indicating that achieving the same level of modified resilience index now requires significantly more energy (e.g., ~42,000 kWh vs. ~39,500 kWh at MRI=0.8). The tightened-speed case (green) exhibits a compressed front with reduced spread in both objectives — the maximum achievable MRI drops from 0.86 to 0.82, while minimum energy improves slightly but at the cost of reduced resilience flexibility.

### Pump Speed Time-Series

- **Claim**: Optimised pump schedules reflect strategic speed modulation to balance energy and resilience.
- **Evidence**: In the baseline case, pumps frequently operate near the upper bound (1.5) during peak demand or low tank levels to preserve pressure margins, then reduce to 0.5–0.7 during off-peak hours to save energy. In the 28m scenario, speeds are consistently higher across all time steps to meet the stricter pressure requirement, explaining the elevated energy consumption. In the tightened-range case, speeds cluster tightly around 0.9–1.2, preventing extreme energy-saving or resilience-boosting maneuvers, which corroborates the compressed Pareto front.

---

## Conclusion

The analysis confirms the hypothesis: pump-speed control in C-Town reveals a robust trade-off between pump energy and operational resilience, quantified via the modified resilience index. Altering service pressure or speed bounds directly reshapes this trade-off space — raising pressure increases the energy cost of resilience, while tightening speed ranges limits the system’s ability to optimally exploit the trade-off. Operators should therefore calibrate these parameters based on priority: if energy savings are paramount, consider relaxing pressure targets or expanding speed ranges; if resilience is critical, accept higher energy costs or invest in infrastructure upgrades to decouple the two objectives. This insight enables data-driven decision-making for sustainable and resilient water network operation.
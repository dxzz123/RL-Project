# Learning when battery energy is worth keeping

The fitted-value controller learns the future opportunity value of the fleet's locations and batteries. It then combines that estimate with the exact current-period dispatch model. PPO remains available as a separate controller for comparison.

The immediate motivation is simple. Recharging spends money now to enable later service. A rule that assigns every remaining kWh the same value can buy too much near the end or miss useful purchases earlier. A learned value can depend on time, location, both batteries, and the other vehicle's state.

This implementation follows the algorithm described in the supplied note. The note's linked ZIP was not included with the attachment, so this is an integration built from its description. The note's numerical results are reference claims; this project's independently generated results are reported below.

## 1. What changes in the controller

The environment's customer choices, physical limits, whole-request assignment, and profit accounting stay as described in [MODEL.md](MODEL.md). The new controller uses a fixed mobile price of **0.60 dollars/kWh** in the main `b=3` configuration. What it learns is **dispatch continuation value**, not a pricing policy.

The current implementation supports **exactly two MCVs**. It stores values for all ordered pairs of vehicle locations and interpolates between battery-grid points. Ordering matters because the simulator allocates requests in vehicle-index order; swapping two vehicle states need not give the same outcome.

For example, 20 additional stored kWh can be valuable early in an episode at a busy location, but have little use after the final customer has departed. The value also depends on whether the other vehicle can serve nearby demand. A single constant inventory coefficient cannot express all of those distinctions.

## 2. The value used by dispatch

Use the same mathematical convention as the manuscript: physical periods are $t=1,\ldots,T$, locations are $l_{m,t}$, and stored battery energies are $e_{m,t}$. Python's period index is `t-1`. The persistent fleet state is

$$
x_t=(l_{1,t},l_{2,t},e_{1,t},e_{2,t}).
$$

The value $\widehat V_t(x_t)$ estimates remaining discounted business profit **before the current period's random requests and choices**, under fixed future pricing. Actual current requests are still observed before dispatch. They are not discarded from the action calculation.

Requests do not carry over between periods in this simulator, and new arrivals and price shocks are sampled independently from prescribed time profiles. Under those assumptions and fixed pricing, time and fleet state are sufficient for this continuation estimate. A model with persistent queues, correlated latent demand, or changing future pricing would need a richer state.

Given the complete current post-choice observation $\widetilde s_t$, the dispatcher chooses

$$
\mathbf a_t^D\in\operatorname*{arg\,max}_{\mathbf a\in\mathcal G(\widetilde s_t)}
\left[
 r(\widetilde s_t,\mathbf a)
 +\gamma\,\operatorname{Interp}\widehat V_{t+1}\bigl(f(\widetilde s_t,\mathbf a)\bigr)
\right].
$$

Here $r$ is the original current-period profit, $f$ returns the next fleet locations and energies, and $\operatorname{Interp}$ interpolates the next-period table. The action grid $\mathcal G$ is the same grid used by the coordinated inventory benchmarks: all individually feasible mode/target combinations, with half and full individually feasible grid-energy requests for recharge and export, before joint station-capacity allocation. Shared station capacity and whole-request allocation are resolved by the existing one-period model.

This search is exact over that finite action grid. It is not continuous-action optimization or a certificate of a globally optimal full-horizon controller. It observes current choices and uses learned expectations about later periods; it receives no future realized random draws.

## 3. How training works backward

After the horizon there is no further operating profit, so

$$
\widehat V_{T+1}(x)=0.
$$

In Python, this terminal table is `values[horizon]`; mathematical $\widehat V_{T+1}$ corresponds to zero-based table index `T`.

The simulator's optional terminal inventory payment, if configured, is already included once in period $T$'s actual reward. It must not be added a second time as continuation value.

For each time and fleet-grid point, the trainer samples $K$ synthetic current-period observations, applies the fixed price and customer-choice model, and computes

$$
\widehat V_t(x)=\frac1K\sum_{k=1}^K
\max_{\mathbf a\in\mathcal G(\widetilde s_t^{(k)})}
\left[
 r(\widetilde s_t^{(k)},\mathbf a)
 +\gamma\,\operatorname{Interp}\widehat V_{t+1}\bigl(f(\widetilde s_t^{(k)},\mathbf a)\bigr)
\right].
$$

The maximum is **inside** the average. Dispatch can adapt to the current observed requests and choices. Maximizing only after averaging would force one action across different observed situations and solve a different problem.

Training starts at the last period, where only immediate profit matters. It then moves backward, using the already-fitted next-period table. Interpolation makes the next-period estimate available when travel or energy losses leave batteries between grid points. Locations are discrete; battery interpolation is bilinear in the two energy coordinates.

This is model-based reinforcement learning, also described as approximate dynamic programming. It uses the known one-period model and arbitrary synthetic fleet-state queries. It has stronger model access than PPO's ordinary episode rollouts. The methodological background is [Bertsekas's approximate dynamic programming chapter](https://web.mit.edu/dimitrib/www/dpchapter.pdf); the implementation here specializes the approach to this simulator.

### What the default budget means

With reserve 5 kWh, capacity 80 kWh, and step 10 kWh, the grid is `[5, 15, 25, 35, 45, 55, 65, 75, 80]`. It includes both physical endpoints. There are 16 ordered location pairs in the four-node city and 81 ordered battery pairs: 1,296 fleet states per time layer.

With 12 periods and 8 samples per period, one training run makes **124,416 fleet-state/draw queries** and stores **16,848 values**, including the terminal layer. Candidate-outcome counts are larger because every query evaluates many joint actions; the actual count and elapsed time are saved in the run summary. These queries are not equivalent to 124,416 on-policy episodes.

Within a period, the same eight random-number seeds are reused across fleet states. This gives common random numbers for comparing states, while independent training seeds generate independent sample banks. Availability, delay, and customer choices are recalculated at each fleet state. Training draws are separate from validation and independent test scenes.

## 4. Train a new model

Activate the existing virtual environment from the project root. Fitted-value training needs the base NumPy environment dependencies; PPO and PPO comparisons additionally need the existing `[train]` extra.

```bash
source .venv/bin/activate
python -m charging_env.rl.fitted_train \
  --scenario configs/utility_b3.json \
  --seed 7 --grid-step 10 --samples 8 \
  --output-dir outputs/fitted_value
```

Omitting `--scenario` uses the original `b=0` toy defaults, so keep the explicit `utility_b3.json` argument for this comparison.

The command prints the actual output directory. Repeating it chooses a numbered sibling automatically, following the same convenience as the PPO CLI. The checkpoint is `checkpoint.npz`; its scenario, fixed-price setting, grid, and value tables are saved together. It uses NumPy arrays and JSON metadata, with pickle disabled when loading.

The trainer rejects unsupported fleet sizes, invalid grids, and excessive configured resource budgets. A coarse grid and eight samples are the starting configuration from the proposal, not a guarantee of good performance in other scenarios.

To use a saved model without training again, load its checkpoint and pass its two decision functions to the common evaluator:

```python
from charging_env.rl.fitted_value import FittedValuePolicy
from charging_env.benchmarks.evaluation import evaluate_policy

policy = FittedValuePolicy.load_checkpoint(
    "outputs/fitted_value_study/fitted/seed_7/checkpoint.npz"
)
result = evaluate_policy(
    policy.scenario,
    [171000001 + 2*i for i in range(64)],
    policy.prices,
    policy.dispatch,
    label="Fitted-value seed 7",
)
print(result["mean_discounted_profit"], result["invalid_actions"])
```

Using `policy.scenario` retains the saved `b=3` scenario and physical settings. These example evaluation seeds are separate from the recorded study; running this snippet does not reproduce its exact test mean.

## 5. Keep the comparison focused

The main comparison contains five methods:

| Method | What it learns or assumes |
|---|---|
| Fitted-value dispatch | Learned continuation table; fixed price 0.60; coordinated current dispatch |
| SA-PPO/MAPPO | Learned node pricing and shared vehicle dispatch; the saved final PPO models |
| Service first | Tuned fixed price and a low-inventory recharge threshold |
| Constant inventory value | Tuned fixed price and one nonterminal stored-energy coefficient |
| Time-dependent inventory value | Fixed price 0.60 and separate early/late coefficients |

The compact comparison does not run random, WAIT, myopic, or adaptive pricing rows. Their older implementations and saved reports remain available for reproducing historical studies.

The three heuristic families are tuned on separate validation scenarios before the final tests. The current grid contains 21 service-first candidates, 45 constant-inventory candidates, and 16 two-phase candidates. The supplied note does not enumerate all of its candidate values, so this project's explicit grid and seeds are recorded as a new protocol rather than claimed to be an exact replay of the note's experiment.

For the constant rule, prices are 0.40, 0.50, 0.60, 0.80, and 1.00; inventory coefficients range from 0.05 through 0.45 in increments of 0.05. For the two-phase rule, the early coefficient is one of 0.10, 0.20, 0.30, 0.40, and the late coefficient is one of 0, 0.05, 0.10, 0.15. The switch occurs halfway through the horizon. Artificial shadow value is zero in the final period; actual terminal reward remains unchanged.

### Reproduce the comparison

Use a new study directory. The study runner preserves fixed paths and protocols so all checkpoints and comparisons can be located reliably. It uses the previous completed 500-update PPO models as frozen controls; it does not retrain or choose a favorable PPO checkpoint after testing.

```bash
python experiments/run_fitted_value_study.py --phase init \
  --study-dir outputs/fitted_value_repeat \
  --ppo-root outputs/b3_improved_study/training
python experiments/run_fitted_value_study.py --phase select \
  --study-dir outputs/fitted_value_repeat
for seed in 7 19 37; do
  python -m charging_env.rl.fitted_train \
    --scenario configs/utility_b3.json --seed "$seed" --grid-step 10 --samples 8 \
    --output-dir "outputs/fitted_value_repeat/fitted/seed_$seed"
done
python experiments/run_fitted_value_study.py --phase evaluate \
  --study-dir outputs/fitted_value_repeat
python experiments/summarize_fitted_value_study.py \
  --study-dir outputs/fitted_value_repeat
python experiments/plot_fitted_value_study.py \
  --study-dir outputs/fitted_value_repeat
```

The last command additionally needs Matplotlib (`python -m pip install matplotlib`). It only plots saved results; it does not train or select another controller. The PPO path must contain the previous study's three final checkpoints; use `--ppo-root` to point to that directory if it has moved.

The comparison uses 128 heuristic-validation scenarios, 512 independent base-test scenarios, and 256 independent scenarios in each of two stress cases. The stress cases increase arrival means by 50% or lower initial inventory to 35 kWh per vehicle. Models and heuristic settings stay frozen. Comparisons are paired within each case; different cases use different scenario sets.

Three fitted-value training seeds and three frozen PPO seeds are executed separately. Their mean rows are equally weighted averages of policy performance, not ensembles. Paired intervals describe scenario uncertainty conditional on these fixed trained policies. Three training seeds give limited evidence about arbitrary retraining. A claim of consistent advantage requires every fitted-value seed to beat the strongest validation-selected heuristic and positive paired uncertainty bounds; the actual predeclared criterion is saved in the protocol.

## 6. Independently generated results

The study completed on **2026-10-20** using `b=3`. Each entry below is mean discounted business profit per episode, in dollars, calculated from the original simulator reward. All methods in a column faced the same scenario seeds. The two learned-method rows average three separately executed models, with training seeds 7, 19, and 37.

| Method | Base: 512 scenarios | High demand: 256 scenarios | Low inventory: 256 scenarios |
|---|---:|---:|---:|
| Fitted-value dispatch | **137.76** | 139.48 | **127.15** |
| SA-PPO/MAPPO | 117.46 | 127.25 | 106.93 |
| Service first | 122.89 | **145.50** | 115.02 |
| Constant inventory value | 129.46 | 130.73 | 117.11 |
| Time-dependent inventory value | 133.39 | 132.84 | 122.94 |

Validation selected the time-dependent rule as the strongest heuristic: price 0.60, early coefficient 0.30, late coefficient 0.15, with the switch at Python period 6. The selected constant rule uses price 0.60 and coefficient 0.25. Service-first uses price 0.80 and recharge threshold 20% of usable battery above reserve (20 kWh stored energy in this scenario). These choices were frozen before testing.

### Consistency across training seeds

Fitted-value dispatch earns **4.37 more than the strongest validation-selected rule**, a **3.28%** improvement. The paired 95% bootstrap interval is **[3.64, 5.11]** dollars. Its gain over PPO is **20.31**, with interval **[19.28, 21.42]**. Both predeclared base-test criteria pass.

| Fitted-value training seed | Base profit | Gain over time-dependent rule | Paired 95% interval for the gain |
|---|---:|---:|---|
| 7 | 137.710 | +4.318 | [3.550, 5.118] |
| 19 | 137.797 | +4.405 | [3.603, 5.198] |
| 37 | 137.788 | +4.396 | [3.588, 5.196] |

The standard deviation of these three model means is 0.048 dollars. This small spread is encouraging, but three seeds do not establish how all future retraining runs will behave. The intervals resample scenarios 5,000 times while holding these trained models fixed; they exclude uncertainty from arbitrary retraining. They also do not imply a win on every individual scenario.

### What operating behavior changes?

Compared with the time-dependent rule, fitted-value dispatch delivers **159.30 versus 140.34 mobile kWh** per base episode. It buys **109.30 versus 80.47 grid kWh** for recharging and finishes with similar fleet inventory: **17.87 versus 16.53 kWh**. The improvement comes from using more energy productively during the episode, rather than leaving a larger battery balance at the end.

The discounted accounting explains the net gain. More mobile service adds 10.34 dollars, lower unmet-demand cost adds 1.81, and export adds 0.08. These gains are partly offset by 4.32 more procurement cost, 1.22 more travel cost, and 2.31 less fixed-station revenue. Together they produce the measured **4.37-dollar** gain. This is a description of the saved operating results, not proof that every additional recharge is profitable.

### Remaining gap under higher demand

Under 50% higher arrival means, **service-first earns 145.50 versus 139.48 for fitted value**. Fitted value's gap is **-6.02**, with paired interval **[-8.67, -3.44]**. The model was trained on base demand and held fixed for this stress test; it does not adapt its continuation table or fixed price to the new demand distribution. The base-case win therefore does not establish an advantage across demand distributions.

In the low-inventory case, fitted value remains ahead of the primary time-dependent rule by **4.20**, with interval **[3.21, 5.18]**. No stress result was used to change the algorithm, choose a new price, or replace the validation-selected primary baseline.

### Budget and saved evidence

The three full fitted-value runs used **373,248 state/draw queries** and evaluated **53,714,325 candidate outcomes** in total. Each run took about 76 seconds on this machine, measured while the three runs executed concurrently. The 96 distinct period/draw seeds per model were reused over grid states; the query count should not be read as that many independent demand samples. PPO is a frozen 500-update control with 96,000 rollout periods per model, trained previously. These methods use different information and training budgets.

The 82 heuristic candidates required **10,496 validation policy episodes**. Final testing executed **9,216 policy episodes** across 1,024 distinct scenarios, with **zero invalid actions**. A separate coarse execution check used 768 state/draw queries and eight development evaluation episodes; those are excluded from the main results. The complete automated suite passed **174 tests**, including interpolation, Bellman calculations, terminal accounting, checkpoint validation, and compact-benchmark checks.

![Learned-value comparison, paired gains, stress results, and an example learned energy value](../outputs/fitted_value_study/fitted_value_comparison.png)

Panel A compares the five base means; dots show the three learned-model means. Panel B shows paired gain intervals against the rule chosen on validation. Panel C includes both stress cases. Panel D illustrates seed 7's learned additional value of battery energy at selected times; it is a fitted training quantity, not an independent profit measurement.

The [study protocol](../outputs/fitted_value_study/protocol.json), [selected rules](../outputs/fitted_value_study/rule_selection/selection.json), [test summary](../outputs/fitted_value_study/benchmarks/summary.json), and [independent numerical audit](../outputs/fitted_value_study/analysis.json) contain exact settings, raw-result paths, counts, and uncertainty estimates. The [executed-source snapshot](../outputs/fitted_value_study/provenance/executed_training_source/) records the training code used. Saved models are under [fitted/seed_7](../outputs/fitted_value_study/fitted/seed_7/), [fitted/seed_19](../outputs/fitted_value_study/fitted/seed_19/), and [fitted/seed_37](../outputs/fitted_value_study/fitted/seed_37/); the three PPO controls remain under [ppo/](../outputs/fitted_value_study/ppo/).

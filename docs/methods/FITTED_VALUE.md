# Original two-vehicle ADP: fitted continuation values

This controller learns **how valuable each fleet state is for the rest of the episode**. At each dispatch decision, it compares feasible actions using today's exact profit plus that learned future value. Earlier studies call this table-based implementation **fitted value**, **fitted-value dispatch**, or **approximate DP**. For different fleet sizes, start with the newer [rollout ADP](GENERAL_ADP.md), which estimates future profit through simulations at decision time.

The controller uses a fixed mobile price of **0.60 dollars/kWh** in the main `b=3` scenario. It learns dispatch, not pricing. It currently supports exactly **two mobile charging vehicles (MCVs)**.

Read [MODEL.md](../MODEL.md) for the operating model, [TRAINING.md](../TRAINING.md) for the common terminal workflow, and [COMPARISON.md](../COMPARISON.md) for the current comparison with both PPO implementations and the important rules. This guide explains how approximate DP works and where to find it in the code.

## Why future battery value matters

Suppose a vehicle can either serve customers now or travel to a fixed charging station (FCS) to recharge. Service earns revenue immediately. Recharging costs money immediately, but may make more profitable service possible later. Comparing only this period's profit can therefore miss a useful recharge.

An inventory rule addresses this by assigning a dollar value to remaining battery energy. A constant rule gives every stored kWh the same value; a time-dependent rule changes that value as the episode progresses. Approximate DP learns a more detailed value that depends on **time, both vehicle locations, and both batteries**. Energy can be useful early at a busy node and almost worthless near the end of the horizon.

The distinction from PPO is direct. PPO learns an actor that outputs action probabilities. This method learns a value table and still searches feasible actions when it operates. The table helps compare their future consequences.

## What the value table represents

Use physical periods $`t=1,\ldots,T`$. Python uses `period = t - 1`. Let

```math
x_t=(l_{1,t},l_{2,t},e_{1,t},e_{2,t}),
```

where $`l_{m,t}`$ is vehicle $`m`$'s location and $`e_{m,t}`$ is its stored energy. The fitted table $`\widehat V_t(x_t)`$ estimates remaining discounted business profit **before that period's random requests and customer choices**, assuming the controller continues to use its fixed price.

The table keeps the vehicles in their original order. This matters because the simulator assigns whole requests in vehicle-index order: exchanging vehicle 1's and vehicle 2's states need not produce the same outcome.

Only the estimate of **future** profit is compressed into this small state. The current dispatch decision still sees all current requests and customer choices. Requests do not carry into later periods, and the simulator samples each period's arrivals and price shocks from its prescribed profiles. With fixed future pricing, those assumptions make time and fleet state sufficient for this continuation estimate. Persistent queues or correlated hidden demand would require a richer state.

Locations are discrete. Batteries use a grid, and values between grid points are obtained by bilinear interpolation: a weighted average of the four surrounding battery-pair values at the same pair of locations.

## How one dispatch decision is made

After customers choose, let $`\widetilde S_t`$ denote the full dispatch observation. The dispatcher evaluates

```math
\mathbf a_t^D\in\mathop{\mathrm{arg\,max}}\limits_{\mathbf a\in\mathcal G(\widetilde S_t)}
\left[
 R_t^d(\widetilde S_t,\mathbf a)
 +\gamma\,\mathrm{Interp}\,\widehat V_{t+1}
 \bigl(f(\widetilde S_t,\mathbf a)\bigr)
\right].
```

Each part has a concrete role:

| Term | Meaning |
|---|---|
| $`\mathcal G(\widetilde S_t)`$ | Feasible joint action candidates for both vehicles |
| $`R_t^d`$ | This period's original business profit, including costs and penalties |
| $`f(\widetilde S_t,\mathbf a)`$ | The next fleet locations and stored energies under that action |
| $`\widehat V_{t+1}`$ | Estimated profit available after this period |
| $`\gamma`$ | Discount applied once between physical periods |

For illustration, an action earning 8 dollars now with an estimated 20-dollar continuation value has score $`8+0.99\times20=27.80`$. Another action earning 12 dollars now but leaving a 10-dollar continuation value has score $`21.90`$. The controller chooses the first. These numbers illustrate the calculation; they are not recorded experimental results.

The candidate grid includes the simulator's feasible modes and destinations. For recharge and export, it requests either **half or all of the individually feasible grid energy**. The original simulator then resolves shared station limits and whole-request allocation. The same candidate generator is used by the coordinated inventory rules, so replacing the inventory coefficient with a learned table does not change the underlying physics.

The action search is exact **over this finite grid**. It is not an optimization over every continuous energy quantity, and the learned continuation values are approximate. The resulting policy is therefore not a certified optimum of the full problem. It uses current observed choices and expected future conditions; it never receives future realized requests.

## How the table is learned

Training works backward, starting at the last physical period. There is no operating profit after the horizon, so

```math
\widehat V_{T+1}(x)=0.
```

At each earlier time and fleet-grid state, the trainer performs four steps:

1. Set the simulator to that time, vehicle-location pair, and battery pair.
2. Sample current requests and prices, quote the fixed mobile price, and simulate customer choice.
3. Find the best dispatch score using current profit and the already-computed next-period table.
4. Repeat for $`K`$ draws and average the best scores.

The resulting backup is

```math
\widehat V_t(x)=\frac1K\sum_{k=1}^K
\max_{\mathbf a\in\mathcal G(\widetilde S_t^{(k)})}
\left[
 R_t^d(\widetilde S_t^{(k)},\mathbf a)
 +\gamma\,\mathrm{Interp}\,\widehat V_{t+1}
 \bigl(f(\widetilde S_t^{(k)},\mathbf a)\bigr)
\right].
```

The maximum is inside the average because dispatch sees the current draw before acting. A different action can be best in each sampled situation. Once every state at a time layer has been fitted, training moves to the preceding layer. This is one backward pass through the finite horizon, not repeated PPO-style updates.

The method is **model-based approximate dynamic programming**. It can query the known simulator from arbitrary fleet states and calculate the outcomes of candidate actions. This is stronger access to the model than PPO's ordinary episode rollouts. Its three sources of approximation are the finite battery grid, the finite action grid, and the sampled expectation.

The optional terminal inventory payment is already included in period $`T`$'s original reward. The zero continuation layer does not remove that payment, and the trainer does not add it twice. In Python, mathematical $`\widehat V_{T+1}`$ is `values[horizon]`.

### What the default training budget means

The `b=3` scenario has four nodes, two vehicles, 12 periods, reserve 5 kWh, and battery capacity 80 kWh. With `--grid-step 10`, the energy grid is `[5, 15, 25, 35, 45, 55, 65, 75, 80]`.

| Quantity per model | Default size |
|---|---:|
| Fleet states per time layer: $`4^2\times9^2`$ | 1,296 |
| Stored values, including the terminal layer: $`13\times1,296`$ | 16,848 |
| State/draw queries: $`12\times1,296\times8`$ | 124,416 |
| Distinct period/draw seeds: $`12\times8`$ | 96 |

Within each period, the same eight random-number seeds are reused across fleet states to make state comparisons less noisy. Availability, waiting times, and customer choices are recomputed for each state. Thus 124,416 queries are neither 124,416 independent demand draws nor 124,416 episodes. Each query also evaluates many joint action candidates; the actual count is saved in `summary.json`.

A finer grid or more samples may improve the approximation but increases work. The present table and exhaustive joint action search are deliberately small. Adding many vehicles would require a different representation or search strategy.

## Train and inspect a model

From the project root, activate the existing virtual environment and run:

```bash
source .venv/bin/activate
python -m charging_env.rl.fitted_train \
  --scenario configs/utility_b3.json \
  --seed 7 --grid-step 10 --samples 8 \
  --output-dir outputs/fitted_value
```

Fitted-value training needs the base NumPy dependencies, not PyTorch. The `[train]` extra is needed when running either PPO or a comparison that includes PPO. See [TRAINING.md](../TRAINING.md) for setup.

Keep `--scenario configs/utility_b3.json`: omitting it selects the original `b=0` defaults. The command prints the actual output directory. If the requested name exists, the CLI automatically uses a numbered sibling such as `outputs/fitted_value_2`.

| Saved file | Use |
|---|---|
| `checkpoint.npz` | Fixed price, scenario, battery grid, and learned value tables |
| `training_metrics.csv` | One row for each fitted time layer |
| `summary.json` | Resource counts, elapsed time, and checkpoint checks |
| `protocol.json` | Exact configuration, sample seeds, and source hashes |

To evaluate the saved study's seed-7 policy on additional scenarios, run this Python example from the project root:

```python
from charging_env.rl.fitted_value import FittedValuePolicy
from charging_env.benchmarks.evaluation import evaluate_policy

policy = FittedValuePolicy.load_checkpoint(
    "outputs/paper_ppo_study/fitted/seed_7/checkpoint.npz"
)
result = evaluate_policy(
    policy.scenario,
    [193000001 + 2*i for i in range(128)],
    policy.prices,
    policy.dispatch,
    label="Approximate DP, seed 7",
)
print("Mean discounted profit:", result["mean_discounted_profit"])
print("Invalid actions:", result["invalid_actions"])
```

Loading `policy.scenario` preserves the training scenario. These example seeds differ from the recorded comparison, so the resulting mean need not match that table. To assess superiority, use the same independent scenarios for every compared controller; [COMPARISON.md](../COMPARISON.md) explains the completed experiment and its limitations.

## Results and implementation references

In the latest shared base test, approximate DP earns **137.76 dollars per episode**, above Joint PPO's **123.19** and earlier PPO's **117.42**. It also leads under low initial inventory, while service-first is stronger under high demand. The current results, uncertainty intervals, and fairness qualifications are maintained in [COMPARISON.md](../COMPARISON.md).

The first fitted-value study completed on **2026-10-20** and used different test scenarios. Its [archived report](../archive/FITTED_VALUE_STUDY.md) preserves the historical results, evidence, and reproduction commands. Use those instructions if you need to build the approximate-DP and earlier-PPO control directory for the current six-method study. Do not mix the two studies' test means.

| Code or test | What to read there |
|---|---|
| [fitted_value.py](../../src/charging_env/rl/fitted_value.py) | State sampling, interpolation, dispatch, backward fitting, checkpoints |
| [fitted_train.py](../../src/charging_env/rl/fitted_train.py) | Training CLI and automatic output-directory naming |
| [policies.py](../../src/charging_env/benchmarks/policies.py) | `joint_dispatch_outcomes`, shared candidate grid and one-period outcomes |
| [test_fitted_value.py](../../tests/test_fitted_value.py) | Bellman backups, interpolation, terminal accounting, and checkpoint validation |

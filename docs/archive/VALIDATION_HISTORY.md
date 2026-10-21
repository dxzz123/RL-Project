# Historical validation record

> Historical record through 2026-10-21. For the current checks and their scope, use [VALIDATION.md](../VALIDATION.md); for performance, use [COMPARISON.md](../COMPARISON.md).

This page records how the project was checked at successive development stages. There are several separate questions: does the simulator follow the intended physical and customer rules, does training execute correctly, can saved results be reproduced, and does the learned policy actually earn good profit? Passing one kind of check does not answer all the others. For example, a saved model may reload perfectly while still earning less than a simple rule.

The latest recorded complete **test suite**, meaning all automated checks run together, has **211 passing tests**. The latest study trained the new manuscript's joint PPO with seeds 7, 19, and 37 and compared it with earlier PPO, fitted value, and three heuristic families. Every new PPO seed beat its earlier-PPO counterpart on the base test. Inventory rules and fitted value still earned higher base profit. Correct execution and good operating decisions are therefore checked separately.

On 2026-10-20, three additional CLI checks brought the suite from 142 to 145 tests by verifying that repeated PPO commands preserve earlier results and use separate destinations. The fitted-value implementation then added 19 learning/checkpoint checks and 10 compact-benchmark checks, bringing the total to 174. Historical study counts below remain records of their earlier stages. The new manuscript PPO then added 29 learning, interface, and checkpoint tests, plus eight study-protocol guards, bringing the total to 211.

[MODEL.md](../MODEL.md) describes the intended behavior, and [ENVIRONMENT.md](../ENVIRONMENT.md) shows how to inspect one simulator step. [Earlier PPO reference](../LEGACY_PPO.md) explains the learning terms used below. In training, a **validation set** specifically means scenarios used to monitor performance and choose checkpoints; this page covers the broader task of checking the implementation and its recorded results.

## 1. Run the checks yourself

From the project root, activate the existing virtual environment. Install the training dependencies if they are not already installed:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
```

A successful unittest run ends with `OK`. For visible examples of the physical model and Gymnasium interface, run:

```bash
python examples/manual_walkthrough.py
python examples/fcs_discharge_walkthrough.py
python examples/check_gym.py
```

The manual example writes `outputs/manual_episode.csv`; it demonstrates a fixed action sequence rather than an optimized policy. The controlled export example has a hand calculation in [ENVIRONMENT.md](../ENVIRONMENT.md#5-check-a-transition-by-hand).

A short training execution check is a separate operation:

```bash
python -m charging_env.rl.train \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir "outputs/validation_check_$(date +%Y%m%d_%H%M%S)"
```

Use a fresh destination. This command checks that learning, saving, and loading execute. Its small number of training episodes cannot establish reliable profit improvement or show that learning has settled to a good solution.

## 2. What the tests verify

### The physical and customer model

Simulator tests check prices chosen from a fixed list at each node, customer-choice probabilities including the option to leave, different energy needs for individual customers, random arrival counts with no artificial cap, random prices at each station, and delay estimates made before customers choose. They also check the information vehicles receive about one another, allocation without serving a customer twice, battery and station limits, balanced energy accounting, profit calculations, and discounting once per physical period.

Some tests deliberately choose customer assignments to check difficult cases, such as several vehicles competing for limited capacity. These specially prepared inputs are called **test fixtures**. They do not come from the usual random customer-choice model. Passing such a test checks that the code handles the chosen case; it does not tell us how often that case would occur in normal operation.

### The Python interface

The Gymnasium checker verifies that observations and actions have the declared formats and allowed values, including a request table whose number of rows can change. It does not show that any PPO library can process that table or correctly handle two decisions per physical period. Separate trainer tests check the recording and processing of these episodes.

A warning about an **unbounded observation** means the model does not assign that quantity a fixed upper limit. Random customer counts and normally distributed price changes have this property. Counts use `int64` computer storage without an artificial demand cap. Gymnasium can also warn that recharge/export actions are measured in actual kWh rather than scaled fractions. These warnings describe interface choices; they do not by themselves indicate a failed physical calculation. [ENVIRONMENT.md](../ENVIRONMENT.md) explains the formats.

### Learning calculations and saved models

Training tests check when future profit is discounted, that credit calculations stop between episodes, that actors use allowed-action masks, that recorded action probabilities can be calculated again, that learned parameters change during updates, and that checkpoints save and restore training state. Recalculating action probabilities is called **likelihood replay**; it checks the numbers used to compare the old and new policies.

With GAE parameter $\lambda_{\mathrm{GAE}}=1$, the targets for both critics equal the full discounted return from the current period through the end of the episode. At zero, they instead use the current period's reward plus a discounted critic prediction from the next pricing state, with no future prediction after the episode ends. Using a critic prediction for later outcomes is called **bootstrapping**. The pricing target still includes customer choice and dispatch in the current period. The symbol $\lambda_{i,t}$ means average customer arrivals and is unrelated to this learning parameter; see [NOTATION.md](../NOTATION.md).

Other tests check the probability of choosing all node prices together, exclusion of nodes without mobile service, vehicle action and energy-fraction probabilities, empty and large request lists, and how request features are combined for the networks. They check that vehicle observations are correct, policies stay fixed during collection, all four networks update, and saved numerical results remain valid.

Later tests cover the optional activation and normalization settings, explicit vehicle identity, the training-only inventory signal, final-period handling, exploration schedules, and limits on gradient size. They also check that imitation examples remain separate from PPO data and that experiment scripts reject inconsistent settings or overlapping data splits. [Earlier PPO reference](../LEGACY_PPO.md) defines these options and explains their purpose.

Checkpoint tests check that loading reproduces evaluation and restores the optimizer history and random-number state needed to continue training. This is **reproducibility**: repeating the recorded calculation from its saved setup. It does not establish that the policy has found the most profitable decisions.

## 3. Latest operational and performance evidence

### Manuscript joint PPO, 2026-10-21

The controller follows the new manuscript's interleaved advantages, one fleet-level PPO ratio, summed fleet entropy, four separate networks and optimizers, and masked discrete dispatch. Both the pricing actor and critic omit realized requests. Equation (11) in the supplied PDF already uses dispatch advantage A2, which is the main target; an optional experimental A1 target is retained separately.

New tests verify the two lambda factors across a physical period, terminal boundaries, joint clipping counterexamples, entropy summation, information restrictions, empty-demand fallback actions, and physically feasible energy requests. They also verify changes in all four networks, deterministic and stochastic checkpoint replay, exact optimizer continuation, RNG isolation, repeat-command destinations, and study source/selection guards. The full **211-test suite passed in 3.560 seconds**.

Three fresh 500-update runs collected **288,000 physical periods**. Each completed all six epochs per update with finite metrics. Monitoring executed 4,224 policy episodes, and saved checkpoints replayed another 192. The three heuristic families were tuned on 128 separate validation scenarios using 82 candidates, or **10,496 policy episodes**. Final testing executed **12,288 policy episodes** across 512 base scenarios and 256 scenarios in each of two stress cases. Both rule validation and final testing recorded **zero invalid actions**.

On the base test, new PPO earned **123.19** versus earlier PPO at **117.42**. The paired improvement was **5.77**, with 95% interval **[4.95, 6.55]**; all three individual seed intervals were positive. New PPO remained below the time-dependent rule at **133.41** and fitted value at **137.76**. Only the predeclared criterion for improvement over earlier PPO passed. Low-inventory seed 19 declined, so the base consistency finding does not apply across every stress condition.

The independent audit recalculates paired uncertainty, actual-model risks, business-profit ledgers, operating metrics, and budgets from saved episodes and checks the original PDF, code snapshot, protocols, and checkpoint hashes. [PAPER_PPO.md](../PAPER_PPO.md) explains the details, operating gap, curves, commands, and limits. The [211-test log](../../outputs/paper_ppo_study/logs/unit_tests.log), [independent audit](../../outputs/paper_ppo_study/analysis.json), and [executed-source snapshot](../../outputs/paper_ppo_study/provenance/executed_training_source) support this record. Small execution checks are recorded separately and were not used to select the main method.

### Fitted-value dispatch, 2026-10-20

The new controller learns a table of future profit for time, fleet locations, and battery levels, then uses the original one-period model to choose coordinated dispatch. Its three models use fixed price 0.60 and the `b=3` scenario. All 12 backward time layers completed with finite values, and loading each saved checkpoint reproduced its value tables exactly. Tests check bilinear interpolation, the maximum inside the sampled Bellman average, discounting once, terminal inventory paid once, ordered vehicle states, and rejection of malformed or incompatible checkpoints. Separate checks confirm the compact rule grid, valid actions, and shared one-period dispatch outcomes.

The three full runs made 373,248 fleet-state/draw queries and evaluated 53,714,325 candidate outcomes. These use a known model and arbitrary synthetic fleet states; they are not on-policy episodes or an equal-information training budget with PPO. The three PPO controls are the previous targeted study's final 500-update models, copied and checked without retraining or selecting a different checkpoint.

The rules were selected on 128 fresh validation scenarios, using 82 candidates and **10,496 policy episodes**. Final evaluation used 512 independent base scenarios and 256 scenarios in each of two stress cases. Three fitted-value models, three PPO models, and three selected rules produced **9,216 test policy episodes** over 1,024 distinct scenarios. Training draw seeds, validation seeds, and all final test sets are separate. Both validation and test evaluations recorded **zero invalid actions**.

Base mean discounted profit was **137.76** for fitted value, **133.39** for the strongest validation-selected rule, and **117.46** for PPO. The fitted-value gain over that rule was **4.37**, with paired 95% bootstrap interval **[3.64, 5.11]**. All three training seeds had positive individual gain intervals, so the predeclared consistency criteria passed. Intervals describe scenario uncertainty for fixed trained models; three seeds provide limited evidence about arbitrary retraining.

The stress tests preserve an important limitation. Fitted value earned **139.48** under higher demand, while service-first earned **145.50**. In the low-inventory case fitted value earned **127.15**, above the primary rule's **122.94**. These tests did not trigger more tuning. See [FITTED_VALUE.md](FITTED_VALUE_STUDY.md) for the full five-method table, operating explanation, budgets, and reproducible commands.

The [saved 174-test log](../../outputs/fitted_value_study/logs/unit_tests.log), [independent numerical audit](../../outputs/fitted_value_study/analysis.json), and [executed-source snapshot](../../outputs/fitted_value_study/provenance/executed_training_source) support this record. The audit independently recomputes profits, paired intervals, risk statistics, and business-reward accounting from the individual episode records and verifies the recorded source/checkpoint hashes. A separate coarse training check and its eight development evaluation episodes are recorded outside the main test budget.

### Earlier targeted PPO study

The targeted `b=3` study finished on 2026-10-18. All three fresh training runs, with seeds 7, 19, and 37, reached 500 updates and 96,000 physical periods each. Metrics were finite, evaluation actions were valid, and loading the checkpoints reproduced evaluation. The selected method used no imitation learning. The separately tested development candidates, including the rejected imitation-assisted candidate, are included when reporting the amount of simulation and training used.

The alternative rules' parameters were selected again on 128 validation scenarios, requiring 7,808 rule-policy episodes. Final evaluation used 512 fresh base scenarios plus 256 fresh scenarios for each of two tests with changed operating conditions, requiring 16,896 policy-episode executions. A **policy-episode execution** means running one policy on one scenario; several policies can run on the same scenario. These totals are therefore not 16,896 distinct scenario draws. All these evaluations recorded zero invalid actions.

On the fresh base test, new average discounted profit is **116.72**, previous-policy profit is **115.93**, and the inventory rule earns **125.51**. The interval for new-minus-previous profit is **[0.10, 1.48]**, calculated from differences between policies on the same scenarios. This is a **paired comparison**. However, seed 7 declines by **2.49**. The new-minus-rule interval is **[-10.03, -7.59]**, entirely below zero. Both requirements specified before testing therefore fail: consistent improvement across all training seeds, and consistent advantage over the rule. The intervals describe scenario-sampling uncertainty for these trained models, rather than variation from arbitrary retraining. Small fluctuations late in training do not establish a strong controller.

An independent numerical review recomputed all 24 reported paired bootstrap comparisons. **Bootstrap** means repeatedly resampling the recorded scenarios to estimate uncertainty in the average comparison. The review also checked counts and file **hashes**, fingerprints used to detect whether source files or the nine checkpoints had changed. A read-only copy preserves the training source exactly as run. Later script changes add stricter checks on reproduction settings without changing learning code. Saved files were checked by their hashes after copying into the project, and 12 saved test episodes were reproduced exactly there.

See [B3_TRAINING_IMPROVEMENTS.md](B3_TRAINING_IMPROVEMENTS.md) for the complete protocol, behavioral gap, plots, and budgets. The [saved 142-test log](../../outputs/b3_improved_study/reviews/unit_tests.log) and [machine-readable analysis](../../outputs/b3_improved_study/analysis.json) support that study's record.

## 4. How the evidence developed

The following test counts record the suite at each development stage. Smaller earlier counts do not contradict the latest total. Each report retains results from the customer model and scenarios used at that time.

| Stage | Tests passing then | What was added or learned | Detailed record |
|---|---:|---|---|
| Environment revision | 38 | Simulator and Gymnasium checks; examples and physical stress cases | Section 5 below |
| First trainer | 54 | Two-stage learning and checkpoint checks; the short run's profit fell | Section 6 below |
| Longer training and monitoring | 64 | Ten monitoring checks; longer runs earned positive profit but did not establish a generally better hyperparameter choice | [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) |
| Frozen-policy benchmarks | 81 | Seventeen benchmark checks; historical `b=0` RL remained below the inventory rule | [POLICY_BENCHMARKS.md](POLICY_BENCHMARKS.md) |
| Dispatch activation pilot | 90 | Nine activation checks, including legacy loading and exact continuation | [ACTIVATION_EXPERIMENT.md](ACTIVATION_EXPERIMENT.md) |
| Charging-benefit extension | 108 | Choice, numerical stability, mask, checkpoint, benchmark, and scenario-routing checks | [OUTSIDE_OPTION.md](../OUTSIDE_OPTION.md) |
| First SiLU + `b=3` study | 112 | Four experiment guards; three 500-update policies still trailed retuned inventory dispatch | [SILU_B3_STUDY.md](SILU_B3_STUDY.md) |
| Targeted `b=3` training study | 142 | Thirty role, normalization, shaping, training, imitation, and experiment checks | [B3_TRAINING_IMPROVEMENTS.md](B3_TRAINING_IMPROVEMENTS.md) |
| Automatic CLI run directories | 145 | Three checks for repeated commands, artifact preservation, and checkpoint continuation | [Earlier PPO reference](../LEGACY_PPO.md) |
| Fitted-value dispatch and compact comparison | 174 | Nineteen value-learning/checkpoint checks and ten compact-rule checks; all three seeds beat the primary rule on the base test | [FITTED_VALUE.md](../FITTED_VALUE.md) |
| Manuscript joint PPO | 211 | Twenty-nine PPO/interface tests and eight protocol guards; all three base seeds beat earlier PPO, but stronger inventory methods remain ahead | [PAPER_PPO.md](../PAPER_PPO.md) |

The longer-training study included a **control**, a comparison run retaining the earlier settings, trained for 48,000 physical periods. Three runs with adjusted settings each used 96,000 periods. On 256 independent scenes, the adjusted policies averaged 16.309 discounted profit, with training-seed standard deviation 0.306, compared with their initial mean of -37.060. Here standard deviation describes the spread across the separate training seeds. Comparisons using equal amounts of training data limit how confidently the gain can be attributed to the new settings.

The historical `b=0` benchmark study completed 3,904 rule-validation episodes and 7,680 base/stress test episodes with valid actions and unchanged checkpoint hashes. Its mean final stochastic RL profit was 16.49 versus the selected inventory rule's 22.75.

The charging-benefit extension included a check over 384 fixed physical states and 5,334 requests, plus two-update Tanh and SiLU execution checks with `b=3`. Keeping the customer situation fixed shows that a larger charging benefit reduces the probability of leaving when charging is available, while preserving relative preferences among charging providers. It does not establish better profit once policies act over whole episodes.

The first SiLU + `b=3` study completed 500 updates per seed, reusing an existing seed-7 run after checking its recorded settings. It executed 7,808 rule-validation and 16,896 held-out test policy episodes without invalid actions. **Held-out** means those test scenarios were reserved from training and model selection. Average final profit was 116.42 versus inventory dispatch at 125.26; the paired gap was -8.84 with interval [-10.06, -7.58]. The targeted PPO study above trained all three seeds afresh and used different test scenarios, so its numbers do not replace this earlier record.

## 5. Historical environment execution record

The environment revision was checked locally on 2026-10-17 using Python 3.12.4, NumPy 2.5.3, and Gymnasium 1.4.0 in the existing `.venv`. All 38 tests passed, including empty and terminal request-sequence checks. All three environment examples passed.

The seeded manual sequence completed 12 periods with total profit -17.31 and discounted profit -16.22. Those numbers describe that illustrative action sequence only. The controlled discharge example matched the hand calculation: each MCV travels 4 km, uses 1.4 stored kWh for travel, exports 10 grid-side kWh, and finishes with 43.073684 stored kWh. Across two vehicles, export revenue is 2 dollars, travel cost 2 dollars, operating cost 1 dollar, and profit -1 dollar.

An additional 250 episodes, or 3,000 physical periods, checked request assignment and energy accounting: 100 varied the distribution of random price changes, and 150 used additional settings and deliberately chosen customer assignments to test difficult limits. This environment-only record predates reinforcement-learning training.

## 6. Historical first-training execution record

The first trainer was checked on 2026-10-17 using Python 3.12.4, NumPy 2.5.3, Gymnasium 1.4.0, and PyTorch 2.14.1 on an arm64 CPU. Its 54-test suite comprised 38 environment tests and 16 training tests. Tested direct dependency versions are recorded in `requirements-training-tested.txt` at the project root.

The saved short run used the following command. Its destination already contains that record; use a different directory when repeating it.

```bash
python -m charging_env.rl.train \
  --updates 10 --episodes-per-update 4 --epochs 4 --seed 7 \
  --eval-episodes 4 --output-dir outputs/sa_ppo_mappo_smoke
```

It completed 40 episodes, or **480 physical periods and 960 stage transitions**. Learning losses and diagnostics were finite; both action-choosing networks and both return-predicting networks changed their learned parameters in each update. The collected episodes and before/after evaluations recorded no invalid actions. Reloading reproduced the final deterministic evaluation exactly: it chooses the highest-probability discrete actions and the mean energy fraction. Separate tests checked continuation when the policy samples actions randomly.

A separate resume check completed one more update, reaching totals of update 11, 44 episodes, and 528 physical periods, with finite metrics and exact replay. It is not included in the 10-update run's counts.

Before and after evaluation used held-out seeds `15, 17, 19, 21`, separate from even-numbered training seeds:

| Mean episode metric | Before training | After training |
|---|---:|---:|
| Undiscounted profit | -0.521593 | -9.805548 |
| Once-per-period discounted profit | -0.454833 | -8.995123 |

Profit decreased. The run established that the training path executed; it did not establish useful learning. That decline motivated the subsequent analyses and benchmark comparisons linked above.

The unchanged saved record is [outputs/sa_ppo_mappo_smoke](../../outputs/sa_ppo_mappo_smoke), including [summary](../../outputs/sa_ppo_mappo_smoke/summary.json), [metrics](../../outputs/sa_ppo_mappo_smoke/training_metrics.csv), [initial evaluation](../../outputs/sa_ppo_mappo_smoke/evaluation_before.json), [final evaluation](../../outputs/sa_ppo_mappo_smoke/evaluation_after.json), and [checkpoint](../../outputs/sa_ppo_mappo_smoke/checkpoint.pt), alongside its scenario and configuration.

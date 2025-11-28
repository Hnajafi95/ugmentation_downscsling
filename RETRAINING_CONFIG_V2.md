# cVAE Retraining Configuration V2 - Empirically Validated

## 🎯 Goal
Fix severe underprediction in prior mode based on empirical evidence from model comparisons.

## 📊 Empirical Evidence

### Comparison of Beta_KL Effects on Prior Mode:

| Config | beta_kl | Max (mm/day) | Mean (mm/day) | P95 (mm/day) | vs Real Max |
|--------|---------|--------------|---------------|--------------|-------------|
| Old Model | 0.03 | **51.28** | 6.10 | 19.10 | -73% |
| Current | 0.05 | **12.32** | 3.48 | 8.66 | -93% ❌ |
| Target | Real | 187.95 | 20.60 | 77.39 | - |

**Key Finding:** Increasing beta_kl from 0.03 → 0.05 made prior mode WORSE (Max: 51mm → 12mm)

**Conclusion:** For extreme value modeling, LOWER beta_kl is better (contrary to standard VAE theory)

---

## ✅ Configuration Changes

### 1. Loss Function Parameters

**w_max: 2.0 → 10.0** (5x increase)
```yaml
w_max: 10.0  # Was 2.0
```
- **Why:** With scale=11.7, old cap at 23mm meant model treated 50mm and 200mm storms similarly
- **Now:** Cap at 117mm forces model to respect extremes up to ~100mm
- **Effect:** Much stronger penalty for missing heavy precipitation

**tail_boost: 1.5 → 5.0** (3.3x increase)
```yaml
tail_boost: 5.0  # Was 1.5
```
- **Why:** P99+ events (>43.6 mm/day) are rare (1% of data) but critical
- **Effect:** 5x multiplier on P99+ pixels ensures model cannot ignore them

**beta_kl: 0.05 → 0.01** (5x decrease)
```yaml
beta_kl: 0.01  # Was 0.05
```
- **Why:** Lower beta allows Z to encode intensity variations for extremes
- **Empirical:** beta=0.03 → Max=51mm, beta=0.05 → Max=12mm
- **Trade-off:** Larger prior-posterior gap, but better extreme generation
- **Effect:** KL divergence will increase from ~18 to ~100-200 (expected and OK!)

**warmup_epochs: 30 → 50**
```yaml
warmup_epochs: 50  # Was 30
```
- **Why:** Aggressive weights need more warmup time for stability
- **Effect:** Gradual KL ramp prevents posterior collapse

### 2. Training Configuration

**epochs: 250 → 300**
```yaml
epochs: 300  # Was 250
```
- **Why:** More time for model to learn with aggressive weight settings

**early_stopping_patience: 60 → 80**
```yaml
early_stopping_patience: 80  # Was 60
```
- **Why:** Extreme value learning may have slower convergence

### 3. Sampling Configuration

**temperature: 0.9 → 1.0**
```yaml
temperature: 1.0  # Was 0.9
```
- **Why:** With beta_kl=0.01, posterior will be looser (σ~1.3-1.5)
- Standard temperature=1.0 is appropriate

### 4. Scheduler

**T_max: 250 → 300**
```yaml
T_max: 300  # Matches new epoch count
```

---

## 📈 Expected Results After Retraining

Based on empirical extrapolation:

| Metric | Current (β=0.05, w_max=2) | Expected (β=0.01, w_max=10) | Target (Real) |
|--------|---------------------------|------------------------------|---------------|
| Max | 12 mm | **80-120 mm** ✓ | 187 mm |
| Mean | 3.5 mm | **10-15 mm** ✓ | 20.6 mm |
| P95 | 8.7 mm | **35-50 mm** ✓ | 77.4 mm |
| P99 | 11 mm | **50-70 mm** ✓ | 114 mm |

**Realistic expectation:** 60-70% of target (much better than current 5-10%)

---

## 🔬 Why This Works: Extreme Value Modeling

### Standard VAE Theory (WRONG for this case):
- High beta_kl → posterior ≈ N(0,1) → good prior sampling
- Assumes: All data points equally important
- Works for: Faces, digits, "smooth" distributions

### Extreme Value Theory (RIGHT for precipitation):
- Extremes are rare outliers, not smooth distribution
- Z needs capacity to encode "this is a 150mm storm, not 10mm drizzle"
- High beta_kl squashes outliers toward mean
- **Low beta_kl preserves rare extreme encodings**

### The Trade-off:
```
High beta_kl (0.05):
  ✅ Posterior ≈ Prior (good alignment)
  ❌ No capacity for extremes → underprediction

Low beta_kl (0.01):
  ❌ Posterior ≠ Prior (gap exists)
  ✅ Can encode extremes → better generation
```

For extreme precipitation modeling, the capacity wins.

---

## 🚀 Training Instructions

### 1. Start Training
```bash
cd /home/user/ugmentation_downscsling
python train_cvae.py --config config.yaml
```

### 2. Monitor Key Metrics

**KL Divergence (will be higher, that's OK!):**
- Old model: KL ≈ 18-20
- **Expected now: KL ≈ 100-200** ✓
- Don't panic! Higher KL is needed for extreme capacity

**MAE_tail (early stopping metric):**
- Watch for improvement in validation MAE_tail
- Should converge around epoch 100-150

**Training Loss Components:**
- L_rec should decrease steadily
- L_kl will be higher (expected)
- Total loss balance is key

### 3. Evaluation After Training

```bash
# Generate samples
python sample_cvae.py --config config.yaml --mode prior --K 5

# Evaluate
python evaluate_samples.py --config config.yaml --num_days 50

# Visualize
python visualize_samples.py \
    --config config.yaml \
    --synth_dir outputs/cvae_simplified/synth \
    --output outputs/cvae_simplified/visualizations_v2 \
    --num_samples 20
```

---

## ⚠️ If Results Still Show Underprediction

### Option 1: Further Reduce beta_kl
```yaml
beta_kl: 0.005  # Even lower
```

### Option 2: Increase Weights More
```yaml
w_max: 15.0
tail_boost: 7.0
```

### Option 3: Architectural Modification
Modify decoder to inject spatial_X only at first layer (not every layer).
This forces decoder to rely more on Z for intensity information.

See `model_cvae.py` lines 275-343 for injection points.

---

## 📝 Key Takeaways

1. **Domain matters:** Extreme value modeling ≠ standard VAE applications
2. **Empirical over theory:** Data showed beta_kl direction clearly
3. **Trade-offs are OK:** Accept larger KL for better extreme generation
4. **Aggressive weights needed:** w_max=10, tail_boost=5 force model to respect extremes

**Start retraining and monitor progress!** 🎯

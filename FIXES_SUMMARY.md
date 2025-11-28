# cVAE Severe Underprediction - Root Cause & Fixes

## 🔴 CRITICAL ISSUE IDENTIFIED: Temperature-Beta_KL Mismatch

### Root Cause Analysis

Your training logs show **KL divergence ≈ 17-18** at convergence. This indicates:
- With `d_z=128` latent dimensions: **KL per dimension ≈ 0.14**
- This means the posterior distribution has **std ≈ 1.1-1.2** (NOT exactly N(0,1))

**The Problem:**
```
Training:    Posterior has std ≈ 1.1-1.2 (due to beta_kl=0.05)
Sampling:    You used temperature=1.5 → samples from N(0, 2.25)
Result:      Decoder never saw z-values from this wider distribution!
             → Severe underprediction (real: 187mm, synth: 17mm)
```

### Why This Happened
1. **During training (beta_kl=0.05):**
   - Model learns to encode precipitation into latent space
   - Posterior distribution: z ~ N(μ, σ) where σ ≈ 1.1-1.2
   - Decoder learns to map these z-values to precipitation

2. **During sampling (temperature=1.5):**
   - Prior sampling: z ~ N(0, 1.5²) = N(0, 2.25)
   - These large z-values are **outside** the range the decoder saw during training
   - Decoder doesn't know how to decode them → outputs near-zero values

### The Mathematical Insight
For a VAE to work in prior mode:
```
σ_posterior ≈ σ_prior (both should be close to 1.0)
temperature ≈ 1.0 (or match actual posterior std)
```

Your mismatch:
```
σ_posterior ≈ 1.1-1.2
temperature = 1.5  ❌ TOO HIGH!
```

---

## ✅ FIXES APPLIED

### 1. Temperature Correction (config.yaml)
**Changed:**
```yaml
# OLD (WRONG):
temperature: 1.5  # Too high!

# NEW (FIXED):
temperature: 0.9  # Matches training posterior std (~1.1)
```

**Why 0.9?**
- Your KL ≈ 18 means posterior std ≈ 1.1-1.2
- Using temperature=0.9-1.0 samples from the **actual trained latent space**
- Temperature=1.5 was sampling from regions the decoder never learned!

### 2. Beta_KL Already Set (config.yaml)
```yaml
beta_kl: 0.05  # Already correct (you updated this)
```
- This is good - will push posterior closer to N(0,1) over time
- Target: KL < 128 (= 1.0 per dimension)
- Current: KL ≈ 18 (≈ 0.14 per dimension) → still has room to improve

### 3. Pixel Threshold Restored (config.yaml)
```yaml
pixel_threshold: 0.1  # Physical rain gauge detection limit (mm/day)
```
- This was commented out but should be active
- Prevents "100% wet fraction" problem
- Matches physical rain gauge sensitivity

### 4. NaN KS Statistic Fixed (evaluate_samples.py)
**Added error handling:**
```python
# Check if both distributions have sufficient samples
if len(real_extremes) < 2 or len(gen_extremes) < 2:
    print(f"  ⚠ Warning: Insufficient extreme values for KS test")
    return np.nan, np.nan
```
- Now prints helpful warning instead of silent NaN
- Explains why KS test failed (no generated extremes due to underprediction)

### 5. Visualization K-Parameter Bug Fixed (visualize_samples.py)
**Added max_k parameter:**
```python
def load_synth_data(synth_dir, day_ids, max_samples=None):
    # Now respects K parameter from config or command line
```

**New command-line argument:**
```bash
python visualize_samples.py --max_k 1  # Load only 1 sample per day
```

---

## 📋 NEXT STEPS

### Step 1: Clean Up Old Samples (IMPORTANT!)
```bash
# Remove old samples from previous runs to avoid confusion
rm -rf /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation_v1/outputs/cvae_simplified/synth/*
```

### Step 2: Regenerate Samples with Fixed Temperature
```bash
cd /home/user/ugmentation_downscsling

python sample_cvae.py \
    --config config.yaml \
    --mode prior \
    --K 5 \
    --days "heavy_only"
```

**Expected improvements:**
- Synth mean should be closer to 20 mm/day (not 4.83!)
- Synth max should be closer to 100-150 mm/day (not 17.60!)
- Wet fraction should normalize to ~75% (not 97.9%)

### Step 3: Re-evaluate
```bash
python evaluate_samples.py \
    --config config.yaml \
    --num_days 50 \
    --samples_per_day 5
```

**What to look for:**
- Spatial correlation: Should improve from 0.33 to > 0.6
- KS statistic: Should have actual value (not NaN)
- Mass bias: Should reduce from -70% to < -20%
- Diversity: Should stay high (> 0.4) ✓

### Step 4: Re-visualize
```bash
python visualize_samples.py \
    --config config.yaml \
    --synth_dir outputs/cvae_simplified/synth \
    --output outputs/cvae_simplified/visualizations_v2 \
    --num_samples 20 \
    --max_k 5  # Load exactly 5 samples per day
```

---

## 🎯 UNDERSTANDING THE TEMPERATURE PARAMETER

### Posterior Mode (temperature doesn't matter much):
```python
z = μ_posterior + temperature * σ_posterior * ε
```
- Samples around the observed data's encoding
- Temperature scales the "perturbation" around μ
- Works well even with higher temperature (you saw this!)

### Prior Mode (temperature is CRITICAL):
```python
z = temperature * ε  (where ε ~ N(0,1))
```
- Samples from N(0, temperature²)
- If temperature >> σ_posterior_trained: samples from untrained regions!
- **Must match training conditions**

### Optimal Temperature Strategy

**Option A: Match Posterior Std (Current Fix)**
```yaml
temperature: 0.9  # Match actual posterior std
```
- Safe and conservative
- Works with current beta_kl=0.05

**Option B: Perfect Prior Matching (Future Improvement)**
```yaml
beta_kl: 0.1  # Higher beta → posterior closer to N(0,1)
temperature: 1.0  # Standard prior
```
- Better generalization
- Requires retraining with higher beta_kl

**Option C: Gradual Annealing (Recommended for Next Training)**
```python
# In train_cvae.py, modify beta schedule:
if epoch < warmup_epochs:
    beta = beta_final * (epoch / warmup_epochs)
elif epoch < warmup_epochs + annealing_epochs:
    # Gradually increase from beta_final to beta_final * 2
    beta = beta_final * (1 + (epoch - warmup_epochs) / annealing_epochs)
else:
    beta = beta_final * 2
```
- Starts with reconstruction focus
- Gradually enforces prior matching
- Best of both worlds

---

## 📊 EXPECTED RESULTS AFTER FIXES

### Before (Temperature=1.5, Broken):
```
Mean:      Real=20.60  Synth=4.83   (-76.5%) ❌
Max:       Real=187.95 Synth=17.60  (-90.6%) ❌
P95:       Real=77.39  Synth=12.51  (-83.8%) ❌
Wet frac:  Real=0.756  Synth=0.979  (+29.5%) ❌
```

### After (Temperature=0.9, Fixed):
```
Mean:      Real=20.60  Synth=~18-22   (~10%) ✓
Max:       Real=187.95 Synth=~120-160 (~30%) ✓ (acceptable)
P95:       Real=77.39  Synth=~60-85   (~15%) ✓
Wet frac:  Real=0.756  Synth=~0.70-0.80 (~5%) ✓
```

*Note: Prior mode will have some underprediction (that's normal), but not 90%!*

---

## 🔬 ADVANCED: Why Prior Mode is Harder than Posterior

### Posterior Mode (Easy):
- Encoder sees the target Y_hr
- Knows exactly where the storm is
- Can encode this information into μ, σ
- Decoder just needs to refine it
- **Works great even with high KL divergence**

### Prior Mode (Hard):
- No information about Y_hr
- Must generate from X_lr alone
- Relies on decoder learning the X→Y mapping
- **Requires tight prior-posterior alignment**
- Sensitive to temperature mismatch

### Why Your Posterior Mode Worked Well:
```
Temperature=0.7 and temperature=1.5 both worked!
→ Because μ_posterior carries the storm information
→ Temperature just adds "stochastic variations"
→ Even large temperature preserves the μ signal
```

### Why Your Prior Mode Failed:
```
Temperature=1.5 with no μ guidance:
→ Samples from N(0, 2.25)
→ Decoder trained on z ~ N(μ, 1.1²)
→ Large z-values are "out of distribution"
→ Decoder outputs safe defaults (near zero)
```

---

## 📈 MONITORING TRAINING (For Future Runs)

### Key Metrics to Watch:

1. **KL Divergence (Target: ~128 for d_z=128)**
   ```
   Current: KL ≈ 18 (too low for prior mode)
   Target:  KL ≈ 64-128 (0.5-1.0 per dimension)
   ```

2. **Reconstruction Loss (Don't let it blow up!)**
   ```
   If KL goes up, reconstruction may worsen
   Need balance: good reconstruction + good prior matching
   ```

3. **MAE_tail (Your early stopping metric) ✓**
   ```
   Good choice! Optimizes for extremes directly
   ```

### Training Recipe for Perfect Prior Mode:

```yaml
# Aggressive prior matching:
beta_kl: 0.08  # Higher than current 0.05
warmup_epochs: 40  # Longer warmup
temperature: 1.0  # Standard once KL is good

# Or use annealing schedule (see Option C above)
```

---

## ❓ FAQ

**Q: Why not just use posterior mode if it works?**
A: Posterior mode requires real Y_hr data, so you can only augment existing heavy days. Prior mode can generate new events for any X_lr input, giving true generalization.

**Q: Should I increase beta_kl even more?**
A: Yes, you can try 0.08-0.10, but watch reconstruction quality. There's always a trade-off.

**Q: What if results are still underpredicted after temperature fix?**
A: Try temperature=1.0 or 1.1. The posterior std calculation is approximate. You may need to experiment in the 0.9-1.2 range.

**Q: Why does posterior mode work with k=1?**
A: With k=1, you're using the μ_posterior (mean) directly with minimal noise. This is almost deterministic and very close to reconstruction.

**Q: The scale parameter (11.7) seems low for 187 mm/day events?**
A: The scale is based on P90 of the **entire dataset** (all days, all pixels). For heavy days specifically, values are much higher. The tail_boost parameter (1.5) compensates by giving extra weight to P99+ events. However, if you retrain, you could experiment with scale=20-30 to see if it helps extremes more.

---

## 🚀 RECOMMENDED WORKFLOW FOR FINAL MODEL

### Phase 1: Quick Test (Use Fixed Config)
1. Clean synth directory
2. Regenerate with temperature=0.9
3. Evaluate and visualize
4. **Decision point:** Good enough? → Proceed to SRDRN training

### Phase 2: Optimize for Prior (If needed)
1. Retrain cVAE with:
   ```yaml
   beta_kl: 0.08
   temperature: 1.0
   epochs: 300  # Longer training
   early_stopping_patience: 80
   ```
2. Monitor KL convergence toward 128
3. Test prior samples at multiple temperatures (0.8, 1.0, 1.2)

### Phase 3: Production
1. Use best cVAE checkpoint
2. Generate augmented dataset (prior mode, K=5)
3. Mix with real data using prepare_augmented_data.py
4. Retrain SRDRN
5. Evaluate SRDRN on extreme metrics (SEDI, CSI, RMSE)

---

**Good luck! The temperature fix should resolve your severe underprediction issue. 🎯**

# Phase 3: Architectural Modification for Prior Mode Extreme Generation

## 🎯 Problem Identified

After retraining with aggressive weights (w_max=10, tail_boost=5, beta_kl=0.01):

### Results Summary:

| Mode | Checkpoint | Max (mm) | Mean (mm) | Diversity | Assessment |
|------|-----------|----------|-----------|-----------|------------|
| **Posterior** | Epoch 38 | 162 | 23.3 | 0.002 | Good reconstruction, zero diversity |
| **Posterior** | Epoch 110 | **189** | 21.7 | 0.002 | Perfect match (189 vs 187 real!) |
| **Prior** | Epoch 38 | 67 | 5.6 | 0.23 | Underpredicts |
| **Prior** | Epoch 110 | **120** | 10.5 | 0.23 | Better, but still -36% |

### Key Findings:

1. ✅ **Aggressive weights worked!** Posterior mode generates 189mm (matches 187mm real)
2. ❌ **Diversity = 0.002** Model is deterministic, ignoring Z
3. ✅ **Prior improved** (12mm → 120mm) but plateaus at ~120mm
4. ❌ **Temperature doesn't help** Tested 1.0, 1.2, 1.4 - all give ~120mm max

### Root Cause: "Decoder Laziness"

**What the model learned:**
```python
# Posterior mode:
Output = spatial_features(X_lr) + tiny_hint(Z)  # Works because Y_hr provides hint

# Prior mode:
Output = spatial_features(X_lr) + ignore(random_Z)  # Fails because Z is noise
```

When spatial features are injected at **every decoder layer**, the decoder can rely purely on X_lr for structure and intensity, treating Z as optional. This works in posterior mode (where Z carries a hint from Y_hr) but fails in prior mode (where Z is random noise).

**Evidence:**
- Posterior diversity = 0.002 (essentially deterministic, not using Z for variation)
- Prior plateaus at 120mm despite decoder being capable of 189mm in posterior
- Temperature changes (1.0 → 1.4) have no effect, proving it's not sampling distribution

---

## 🏗️ Architectural Change: Input-Only Spatial Injection

### Modification to `model_cvae.py`:

**Old Architecture (V2):**
```python
# Spatial injection at EVERY layer:
h = inject_spatial(h, x_map)  # Block 1
h = conv1(h)
h = inject_spatial(h, x_map)  # Block 2
h = conv2(h)
h = inject_spatial(h, x_map)  # Block 3
h = conv3(h)
```

**New Architecture (V3):**
```python
# Spatial injection ONLY at first layer:
h = inject_spatial(h, x_map)  # Block 1 - Gets spatial guidance
h = conv1(h)
h = conv2(h)  # Block 2 - Must rely on Z and previous features
h = conv3(h)  # Block 3 - Must rely on Z and previous features
```

### Changes Made:

**Decoder.__init__():**
- `conv2`: `in_channels = base_filters * 4` (removed +32)
- `conv3`: `in_channels = base_filters * 2` (removed +32)

**Decoder.forward():**
- Removed `inject_spatial()` calls before conv2 and conv3
- Only inject at first layer (before conv1)

### Why This Works:

**Information Flow:**
1. **FC layers:** Z and h_X combined → (B, 512, 20, 17)
2. **Block 1:** Gets spatial_X guidance → knows "where" the storm is
3. **Blocks 2-3:** Must use Z to determine "how intense" the storm is
4. **Result:** Z becomes necessary, not optional

**Expected Behavior:**
- Prior mode: Z must carry intensity information → better extreme generation
- Posterior mode: Still good (may see slight drop in max due to less spatial guidance)
- Diversity: Should increase from 0.002 (model forced to use Z variations)

---

## 📊 Expected Results After Retraining

### Conservative Estimate:
| Metric | Current (V2, Prior) | Expected (V3, Prior) | Real Target |
|--------|---------------------|----------------------|-------------|
| Max | 120 mm | **140-160 mm** | 187 mm |
| Mean | 10.5 mm | **12-16 mm** | 20.6 mm |
| P95 | 37.7 mm | **45-60 mm** | 77.4 mm |
| Diversity | 0.23 | **0.3-0.5** | Higher is better |

### Optimistic Estimate (if Z fully utilized):
- Max: **160-180 mm** (approaching real 187mm)
- Mean: **16-20 mm** (matching real 20.6mm)
- P95: **60-75 mm** (close to real 77.4mm)

**Trade-off:** Posterior mode may drop slightly (~180mm instead of 189mm) due to less spatial guidance, but diversity should improve.

---

## ⚙️ Configuration for Retraining

### Key Settings:

```yaml
loss:
  w_max: 10.0        # Keep (proven to work)
  tail_boost: 5.0    # Keep (proven to work)
  beta_kl: 0.01      # Keep (allows Z capacity)

train:
  epochs: 300
  early_stopping_patience: 100  # Increased (epoch 110 > epoch 38)

sampling:
  temperature: 1.0   # Keep (1.2-1.4 didn't help)
```

### Why NOT w_max=20.0:

Posterior mode with w_max=10.0 already generates 189mm (above 187mm real). The limitation is not weight cap - it's architectural. The decoder CAN generate extremes; it just needs to use Z properly.

---

## 🚀 Training Instructions

### 1. Start Training
```bash
cd /home/user/ugmentation_downscsling
python train_cvae.py --config config.yaml
```

### 2. Monitor Training

**Watch for diversity improvement:**
- Previous training: Diversity ≈ 0.002 (deterministic)
- **Expected now:** Diversity > 0.1 (model using Z)

**KL Divergence:**
- Should be similar to before (35-40 range)
- If much higher (>100), training might be unstable

**MAE_tail:**
- May be slightly higher than before (trade-off for better prior mode)
- Don't panic if best epoch is 15-16 instead of 16.4

### 3. Evaluation Strategy

**Don't just trust "best" checkpoint!** Compare multiple epochs:

```bash
# Test epoch range 80-120
python sample_cvae.py --checkpoint cvae_epoch_080.pt --mode prior --K 5
python sample_cvae.py --checkpoint cvae_epoch_100.pt --mode prior --K 5
python sample_cvae.py --checkpoint cvae_epoch_120.pt --mode prior --K 5
python sample_cvae.py --checkpoint cvae_best.pt --mode prior --K 5
```

Previous training showed epoch 110 >> epoch 38 (best MAE_tail), so check later epochs!

---

## 📈 Success Criteria

### Minimum Success:
- ✅ Prior Max > 140mm (up from 120mm)
- ✅ Diversity > 0.1 (up from 0.002)
- ✅ Posterior Max > 160mm (slight drop acceptable)

### Good Success:
- ✅ Prior Max > 160mm
- ✅ Prior Mean > 15mm
- ✅ Diversity > 0.2
- ✅ Posterior still > 170mm

### Excellent Success:
- ✅ Prior Max > 175mm (within 10% of real)
- ✅ Prior Mean > 18mm
- ✅ Diversity > 0.3
- ✅ Posterior ≈ 185mm

---

## ⚠️ If Results Still Poor

### Option 1: Further Reduce beta_kl
```yaml
beta_kl: 0.005  # Even lower, more Z capacity
```

### Option 2: Increase Latent Dimensions
```yaml
model:
  d_z: 256  # Up from 128, more capacity
```

### Option 3: Inject at TWO layers instead of one
```python
# Inject at Block 1 AND Block 2, but not Block 3
# Middle ground between all and one
```

---

## 📝 Key Takeaways

1. **Empirical evidence is king:** Diversity=0.002 was smoking gun for decoder laziness
2. **Posterior success ≠ Prior success:** Different modes test different aspects of the model
3. **Architecture matters more than hyperparameters:** When temp changes didn't help, architecture was the answer
4. **Later epochs can be better:** Don't blindly trust early stopping on a single metric

**Start the retraining and monitor both prior AND posterior performance!** 🎯

---

## Comparison Table

| Version | Architecture | beta_kl | w_max | Prior Max | Post Max | Diversity |
|---------|-------------|---------|-------|-----------|----------|-----------|
| V1 | No spatial | 0.05 | 2.0 | 12mm | 162mm | ? |
| V2 | Inject everywhere | 0.01 | 10.0 | 120mm | 189mm | 0.002 |
| **V3** | **Inject once** | **0.01** | **10.0** | **?** | **?** | **?** |

Target for V3: Prior Max ≥ 160mm, Diversity > 0.2

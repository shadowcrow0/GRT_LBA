**Differences Between PS and Non-PS Implementations**

### **Model 1 (PS, 8 parameters) — *Model1_PS_8param.py***

**Core idea:**
Under *Perceptual Separability* (PS), the drift rate for each dimension depends **only** on the stimulus level of that same dimension and is **independent** of the other dimension.

**Only 8 parameters are defined** (lines 925–932):

```python
v1_L_when_H = pm.TruncatedNormal("v1_L_when_H", mu=1.8, sigma=2.0, lower=0.001, upper=5.0)
v2_L_when_H = pm.TruncatedNormal("v2_L_when_H", mu=1.5, sigma=2.0, lower=0.001, upper=5.0)
v1_L_when_V = pm.TruncatedNormal("v1_L_when_V", mu=1.5, sigma=2.0, lower=0.001, upper=5.0)
v2_L_when_V = pm.TruncatedNormal("v2_L_when_V", mu=1.8, sigma=2.0, lower=0.001, upper=5.0)
# and similarly for the Right dimension
```

**Shared parameters enforced by PS constraints** (lines 948–975):

```python
v_tensor = pt.zeros((4, 2, 2))

# Condition 0: HH
v_tensor = pt.set_subtensor(v_tensor[0, 0, 0], v1_L_when_H)  # ← shared!
v_tensor = pt.set_subtensor(v_tensor[0, 1, 0], v1_R_when_H)  # ← shared!

# Condition 1: HV
v_tensor = pt.set_subtensor(v_tensor[1, 0, 0], v1_L_when_H)  # ← same v1_L_when_H!
v_tensor = pt.set_subtensor(v_tensor[1, 1, 0], v1_R_when_V)  # ← uses the V parameter

# Condition 2: VH
v_tensor = pt.set_subtensor(v_tensor[2, 0, 0], v1_L_when_V)  # ← uses V parameter
v_tensor = pt.set_subtensor(v_tensor[2, 1, 0], v1_R_when_H)  # ← same v1_R_when_H!
```

**Key point:**
`v1_L_when_H` is used in both **HH** and **HV** conditions → this *is* the PS constraint.

---

### **Model 2 (Non-PS, 16 parameters) — *Model2_NonPS_16param.py***

**Core idea:**
No PS constraints. Each condition has its own parameters, allowing *cross-dimensional interactions*.

**Defines 16 independent parameters** (lines 935–968):

```python
# Condition 0: HH
v1_L_HH = pm.TruncatedNormal("v1_L_HH", ...)
v2_L_HH = pm.TruncatedNormal("v2_L_HH", ...)
v1_R_HH = pm.TruncatedNormal("v1_R_HH", ...)
v2_R_HH = pm.TruncatedNormal("v2_R_HH", ...)

# Condition 1: HV
v1_L_HV = pm.TruncatedNormal("v1_L_HV", ...)   # ← different variable!
v2_L_HV = pm.TruncatedNormal("v2_L_HV", ...)
# ... and so on, totaling 16 parameters
```

**Each condition uses its own parameters** (lines 973–989):

```python
# Condition 0: HH
v_tensor = pt.set_subtensor(v_tensor[0, 0, 0], v1_L_HH)
v_tensor = pt.set_subtensor(v_tensor[0, 1, 0], v1_R_HH)

# Condition 1: HV
v_tensor = pt.set_subtensor(v_tensor[1, 0, 0], v1_L_HV)  # ← distinct!
v_tensor = pt.set_subtensor(v_tensor[1, 1, 0], v1_R_HV)

# Condition 2: VH
v_tensor = pt.set_subtensor(v_tensor[2, 0, 0], v1_L_VH)  # ← again distinct!
v_tensor = pt.set_subtensor(v_tensor[2, 1, 0], v1_R_VH)
```

**Key point:**
`v1_L_HH`, `v1_L_HV`, `v1_L_VH`, `v1_L_VV` are all independent → **no constraints**.

---

### **Visualization**

**PS Model (8 parameters):**

```
HH: v1_L_when_H ----┐
HV: v1_L_when_H ----┘  (shared)

VH: v1_L_when_V ----┐
VV: v1_L_when_V ----┘  (shared)
```

**Non-PS Model (16 parameters):**

```
HH: v1_L_HH   (independent)
HV: v1_L_HV   (independent)
VH: v1_L_VH   (independent)
VV: v1_L_VV   (independent)
```

---

### **Why this design?**

**Testing the PS hypothesis:**
If the data truly follow PS, then the 8-parameter model should be sufficient → the simpler model should win.

**WAIC penalizes complexity:**
Although the 16-parameter model is more flexible, it will be penalized if the additional parameters are unnecessary.

**Experimental setup:**
Since we generate data *under the PS model*, we expect **Model 1 (PS)** to outperform **Model 2 (Non-PS)**.


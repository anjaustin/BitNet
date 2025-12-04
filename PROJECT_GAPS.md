# Project Gaps Analysis: BitNet Dithering

## Objective
Frontier-quality inference from a 1.58-bit ternary model.

---

## Iteration 1: Initial Gap Identification

### What We Have
- Dithering C++ implementation (Bayer matrix 4x4/8x8, adaptive strength)
- Integration hooks in ggml-bitnet-lut.cpp and ggml-bitnet-mad.cpp
- CMake configuration linking dithering library
- Observed temperature stability up to 174
- Partial Mirostat entropy mapping
- Working inference via ti.py

### What We Don't Have

**Verification Gaps:**
1. No confirmation dithering code is actually executing at runtime
2. No logging or metrics from dithering functions
3. No controlled A/B comparison (dithering enabled vs disabled)

**Measurement Gaps:**
4. No baseline quality benchmarks
5. No comparison against other 2B models (Phi-2, Gemma-2B, etc.)
6. No perplexity measurements
7. No standardized evaluation suite (MMLU, HellaSwag, etc.)

**Understanding Gaps:**
8. Don't know WHY temperature stability improved (dithering? min_p? both?)
9. Don't understand optimal dithering parameters for this model
10. Don't know if dithering helps at inference time or only at quantization time

**Infrastructure Gaps:**
11. No reproducible test harness
12. No version control of parameter configurations
13. No systematic way to compare outputs across runs

---

## Iteration 2: Prioritization

Which gaps block progress vs which are nice-to-have?

**Blockers (must fix to claim anything):**
1. Verify dithering is executing - without this, all observations are suspect
2. A/B comparison - without this, we can't attribute improvements to dithering
3. Baseline benchmarks - without this, "frontier quality" is just a claim

**High Value (significantly advances the work):**
4. Understand causality (dithering vs min_p vs both)
5. Standardized eval suite - enables comparison with published results
6. Reproducible test harness - enables iteration

**Nice to Have (polish, not substance):**
7. Parameter optimization sweeps
8. Comparison with other 2B models
9. Detailed logging infrastructure

---

## Iteration 3: Concrete Actions

For each blocker, what specifically needs to happen?

### 1. Verify Dithering Execution

**Action:** Add logging to `bitnet_apply_ordered_dithering()` that outputs:
- Whether function was called
- Input tensor statistics (mean, std, min, max)
- Output tensor statistics (same)
- Dithering strength applied

**How:** Modify `src/bitnet_dithering.cpp`, add `fprintf(stderr, ...)` or proper logging
**Validation:** Run inference, confirm log output appears

### 2. A/B Comparison (Dithering On vs Off)

**Action:** Create build configurations:
- `BITNET_DITHERING_ENABLED=1` (current)
- `BITNET_DITHERING_ENABLED=0` (control)

**How:** 
- Already have preprocessor guards
- Need two builds: `build_dither/` and `build_nodither/`
- Run identical prompts through both at same temperature
- Compare outputs qualitatively and quantitatively

**Validation:** Observable difference (or lack thereof) in output quality at high temperatures

### 3. Baseline Benchmarks

**Action:** Establish quality baseline using:
- Fixed prompt set (10-20 diverse prompts)
- Fixed parameters (temp, mirostat, etc.)
- Human evaluation rubric (coherence, accuracy, depth)
- Optional: perplexity on held-out text

**How:**
- Create `benchmarks/prompts.json` with test prompts
- Create `benchmarks/evaluate.py` to run prompts and collect outputs
- Manual scoring on 1-5 scale for key dimensions

**Validation:** Reproducible scores that can be compared across configurations

---

## Iteration 4: Dependencies and Ordering

What order should these happen in?

```
1. Verify dithering execution
   └── Must confirm code runs before testing its effects
   
2. A/B comparison build setup
   └── Depends on: verification complete
   └── Produces: two comparable binaries
   
3. Baseline benchmark creation
   └── Can happen in parallel with #2
   └── Produces: evaluation framework
   
4. Run A/B comparison with benchmarks
   └── Depends on: #2 and #3 complete
   └── Produces: evidence for/against dithering benefit
   
5. If dithering helps → optimize parameters
   If dithering doesn't help → investigate why, or pivot
```

---

## Iteration 5: Refined Understanding

Reading back iterations 1-4, what's the core insight?

**We built something we haven't validated.**

The temperature stability is real - we observed it. But we attributed it to dithering without proof. Gemini suggested min_p might be the cause. We don't actually know.

The honest state of the project:
- Integration: DONE (code is in place)
- Verification: NOT DONE (don't know if code runs)
- Validation: NOT DONE (don't know if code helps)
- Optimization: PREMATURE (optimizing unvalidated code)

**The critical path is:**
1. Prove dithering runs
2. Prove dithering helps (or doesn't)
3. THEN optimize/iterate

Everything else is premature.

---

## Iteration 6: Minimum Viable Validation

What's the fastest path to knowing if dithering matters?

**Option A: Full benchmark suite**
- Time: 2-4 hours to set up, hours to run
- Rigor: High
- Overkill for initial validation

**Option B: Single high-temperature comparison**
- Build with dithering, run at temp 50, save output
- Build without dithering, run at temp 50, save output
- Compare qualitatively
- Time: 30 minutes
- Rigor: Low but informative

**Option C: Add logging, observe**
- Add print statements to dithering functions
- Run normal inference
- See if they fire
- Time: 10 minutes
- Rigor: Only validates execution, not effect

**Recommendation:** Do C first (10 min), then B (30 min). If B shows difference, invest in A.

---

## Iteration 7: Convergence

**What needs to happen:**

1. **Immediate (10 min):** Add logging to dithering functions, verify execution
2. **Short-term (30 min):** Build with/without dithering, compare at high temp
3. **If promising (2-4 hrs):** Create proper benchmark suite
4. **Ongoing:** Document findings, iterate on dithering parameters

**What I recommend we do next session:**
- Step 1: Add fprintf to `bitnet_apply_ordered_dithering()` 
- Step 2: Run inference, confirm dithering is called
- Step 3: Create no-dither build
- Step 4: Same prompt, same temp, both builds, compare

This gives us ground truth. Everything else builds on that.

---

## Iteration 8: Critical Realization

Wait. Re-reading the integration...

We added dithering to `quantize_i2_s()`. This function is called during **model quantization**, not during inference. 

This means:
1. Dithering affects the GGUF file when it's created
2. The model we've been testing (`ggml-model-i2_s.gguf`) was quantized BEFORE our integration
3. Our temperature tests were on a model WITHOUT dithering applied

**The dithering code exists but hasn't affected any model we tested.**

To actually test dithering, we need to:
1. Re-quantize the original model with dithering enabled
2. Compare the new GGUF against the original GGUF
3. THEN test inference quality

This changes the validation path entirely.

---

## Revised Convergence

**Actual next steps:**

1. Locate original (non-quantized) model weights
2. Run quantization with dithering enabled, produce new GGUF
3. Run quantization with dithering disabled, produce control GGUF
4. Compare inference from both at various temperatures
5. THEN draw conclusions about dithering's effect

The logging question becomes: does quantization call our dithering code?

---

## Final Assessment

The project has promising observations but unvalidated claims. More critically: **the observations may not involve dithering at all** if the tested model was quantized before integration.

The gap is larger than I initially thought. We need to:
1. Confirm dithering is called during quantization
2. Re-quantize with dithering
3. Then test

The engineering touched the codebase. Whether it touched the model is unknown.

---

*End of gap analysis.*

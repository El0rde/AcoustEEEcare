# Additional Firmware Diagnosis: Persistent Buzzing After Issues 1 & 2 Fixes

**Date:** June 1, 2026  
**Status:** Issues 1 & 2 applied but problem persists  
**Latest Attempt:** DC offset calibration + 40µs acquisition time

---

## Critical Issues Identified in Code

### **Issue 2A: Race Condition in DC Calibration Buffer Reset** ⚠️ CRITICAL
**Location:** `src/main.c`, lines 749-751  
**Current Code:**
```c
calibrate_dc_offset_at_start();

/* Reset staging pointers for actual recording (calibration used them) */
stage_wr = stage_rd = 0;
k_sem_reset(&stage_sem);
```

**Problem:**
- The SAADC ISR is **still running** while we reset `stage_wr` and `stage_rd`
- **Race condition:** If the ISR completes a SAADC_EVT_DONE while we're resetting:
  1. ISR reads `stage_wr` (say, = 5)
  2. We reset `stage_wr = 0` in main thread
  3. ISR increments: `stage_wr = (5 + 1) % 16 = 6`
  4. Main loop reads from `stage_rd = 0`, but buffer position 0 may contain garbage from calibration
  5. **Main loop reads corrupted/uninitialized data**

- This causes the first 1-2 seconds of recording to contain random data mixed with actual audio
- The "buzzing" could be this corrupted data

**Fix:**
The staging buffer should NOT be reset after calibration. Instead:
1. Let the ISR continue writing into the next slots (will wrap around naturally)
2. The main loop starts reading from wherever we left off (not from slot 0)
3. Or: use a separate calibration buffer, not the staging buffer

**Recommended Implementation:**
```c
/* Option 1: Skip resetting, just restart from current positions */
// After calibration ends, stage_wr and stage_rd are valid
// Don't reset them; main loop will read from stage_rd naturally
// This way, no data corruption

/* Option 2: Use a separate 1-slot calibration buffer */
static int16_t calib_buf[HALF_BUF_SAMPLES];
// Modify SAADC to write to this during calibration
// Restore original pointers when switching to recording
```

---

### **Issue 2B: DC Calibration Doesn't Account for Post-Calibration State**
**Location:** `src/main.c`, lines 625-665  
**Current Code:**
```c
stage_wr = stage_rd = 0;
k_sem_reset(&stage_sem);
// ... capture 800 samples ...
```

**Problem:**
- After calibration, we've consumed 800 samples from the staging buffer
- But `stage_wr` has been incrementing the entire time the SAADC was running
- When we reset `stage_wr = 0`, we lose track of which staging slots contain valid data
- The ISR doesn't know about the reset and may overwrite slots the main loop thinks are empty

**Fix:**
Track the state properly or use a separate calibration approach:
```c
// Better approach: don't reset, let ISR continue naturally
uint8_t calib_start_rd = stage_rd;  // Remember where calibration started
// ... calibrate ...
// Now stage_rd has advanced, stage_wr has advanced
// The main loop just continues from the new stage_rd position
// No reset needed!
```

---

### **Issue 3A: DC Offset Might Be Unstable During Calibration**
**Location:** `src/main.c`, lines 627-665  
**Current Code:**
```c
/* Capture calibration samples */
uint32_t samples_captured = 0;
while (samples_captured < calibration_samples) {
    if (k_sem_take(&stage_sem, K_MSEC(200)) != 0) {
        LOG_WRN("DC calibration: timeout...");
        break;  // EXIT WITH PARTIAL DATA!
    }
```

**Problem:**
- If the SAADC is slow or busy, the `k_sem_take` might timeout after 200ms
- But we only expect 800 samples = 100ms at 8kHz
- With 16 staging slots and semaphore based on HALF_BUF_SAMPLES (512), we should only need 2 semaphore signals
- A 200ms timeout might be too aggressive
- **If timeout occurs, `count` could be 0 or very small**, and we fall back to hardcoded 1024
- But if actual DC offset is 2048 (due to gain/divider configuration), using 1024 will remove half the signal

**Evidence of Timeout:**
- If logs show "DC calibration: timeout", the calibration failed
- Fallback to 1024 might be completely wrong

**Fix:**
```c
#define CALIB_TIMEOUT_MS 500  /* Increased from 200ms for 800 samples @ 8kHz */
// Or: measure exact expected time
#define CALIB_SAMPLES 800
#define CALIB_BLOCKS ((CALIB_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES)
#define CALIB_TIMEOUT_MS ((CALIB_BLOCKS * HALF_BUF_SAMPLES * 1000 / SAMPLING_RATE) + 100)

if (k_sem_take(&stage_sem, K_MSEC(CALIB_TIMEOUT_MS)) != 0) {
    // ... handle timeout ...
}

// Add logging to verify calibration succeeded:
if (count < calibration_samples * 0.8) {  // Less than 80% of expected
    LOG_WRN("DC calibration incomplete: got %u/%u samples", 
            (unsigned)count, (unsigned)calibration_samples);
}
```

---

### **Issue 4: SAADC Gain Still GAIN1_4 (Not Applied)**
**Location:** `src/main.c`, line 119  
**Status:** Fix from Issue 4 diagnosis NOT implemented  
**Current Code:**
```c
.gain       = NRF_SAADC_GAIN1_4,
```

**Problem:**
- With GAIN1_4, full-scale input is 3.3V (entire VDD range)
- Signal is centered at 1.65V with swing ±100mV (typical)
- ADC utilizes only ~50% of range (1.55-1.75V out of 0-3.3V)
- **Effective resolution is reduced from 12-bit to 11-bit**
- Quantization noise floor rises, weak signals disappear

**Expected ADC Values with GAIN1_4:**
- 1.55V input → ADC ≈ 1920 (if using expected DC offset 2048)
- 1.75V input → ADC ≈ 2176
- Signal amplitude = 256 counts out of 4096 = 6.25% of range = ~6-bit effective resolution

**Fix:**
Change to GAIN1_2 (full-scale = 1.65V, signal uses full range):
```c
.gain       = NRF_SAADC_GAIN1_2,  // CHANGED from GAIN1_4
```

**Then update DC offset:**
With GAIN1_2, full-scale input is VDD/2 = 1.65V, so:
- DC offset = 2048 (midpoint of 12-bit range)

**Code Changes:**
1. Change `.gain` from `GAIN1_4` to `GAIN1_2`
2. Update fallback DC offset from `1024` to `2048`
3. Update log message in `saadc_init_once()` from "GAIN1_4" to "GAIN1_2"

---

### **Issue 5: DC Removal IIR Filter Convergence Too Slow**
**Location:** `src/main.c`, lines 502-504  
**Current Code:**
```c
dc_estimate += ((int32_t)raw[i] - dc_estimate) >> 8;  // Coefficient = 1/256
dc_copy[i]   = (int16_t)((int32_t)raw[i] - dc_estimate);
```

**Problem:**
- Time constant = 256 samples = 32ms @ 8kHz
- Filter doesn't converge to steady state for ~150-200ms (5 time constants)
- If calibration measured wrong offset, or if there's drift, the filter takes too long to correct
- **During first 150-200ms of recording, DC-removed signal is distorted**
- This causes low-frequency content (bass, voice fundamentals) to be attenuated

**Example Scenario:**
- True DC offset = 2048
- Calibration measures 2100 (slight error)
- Filter starts at 2100
- Error = 2048 - 2100 = -52 ADC counts
- After 32ms: filter has moved only ~20 counts
- After 100ms: filter has moved ~40 counts
- **First 100ms of audio has wrong DC removal**

**Fix:**
Use faster convergence:
```c
// Change >> 8 to >> 4 for faster convergence (1/16 coefficient)
dc_estimate += ((int32_t)raw[i] - dc_estimate) >> 4;  // 64 sample time constant = 8ms
```

**OR:**
Increase initial calibration accuracy so filter starts closer to true offset:
```c
// Verify calibration succeeded with high confidence
if (count < calibration_samples) {
    LOG_WRN("Calibration incomplete, results may be unreliable");
    // Could skip recording or alert user
}
```

---

### **Issue 6: Right-Shift on Signed int16_t May Corrupt Data**
**Location:** `src/main.c`, lines 236-238  
**Current Code:**
```c
for (uint16_t _k = 0; _k < HALF_BUF_SAMPLES; _k++) {
    filled[_k] = (int16_t)(filled[_k] >> 2);
}
```

**Problem:**
- `filled` is `int16_t *` (signed)
- If SAADC returns any negative values (e.g., -4000 from oversample accumulation)
- Arithmetic right-shift on int16_t will **sign-extend**:
  - `-4000 >> 2` = `-1000` (sign bit is extended)
  - Logical right-shift would give different result
- Comment says range is [0, 2047], which suggests unsigned values
- **Type mismatch between expected behavior (unsigned) and actual type (signed)**

**Potential Impact:**
- If SAADC ever returns negative accumulated values, they'll be sign-extended
- Negative post-shift values will remain negative, skewing the signal
- DC removal will then subtract wrong value

**Fix:**
Use unsigned right-shift explicitly:
```c
for (uint16_t _k = 0; _k < HALF_BUF_SAMPLES; _k++) {
    filled[_k] = (int16_t)((uint16_t)filled[_k] >> 2);  // Logical right-shift
}
```

Or verify the SAADC data type is correct:
```c
// Add diagnostic logging
LOG_INF("Raw SAADC samples (diagnostic):");
for (int i = 0; i < 10 && i < HALF_BUF_SAMPLES; i++) {
    // Log before right-shift to see actual SAADC output range
    LOG_INF("  [%d] raw = %d, shifted = %d", i, filled[i], (filled[i] >> 2));
}
```

---

### **Issue 7: Overconfidence in DC Offset Fallback Value**
**Location:** `src/main.c`, line 658  
**Current Code:**
```c
} else {
    dc_estimate = 1024;  /* Fallback to default */
    LOG_WRN("DC offset calibration failed, using fallback: 1024");
}
```

**Problem:**
- If calibration fails (timeout, no data), we fall back to 1024
- But if you change GAIN to GAIN1_2 (Issue 4), the correct fallback should be 2048
- Using wrong fallback will remove half the signal

**Fix:**
Use a #define for fallback and update it when gain changes:
```c
#if SAADC_GAIN_USED == NRF_SAADC_GAIN1_4
#define DC_OFFSET_FALLBACK 1024
#else  /* GAIN1_2 */
#define DC_OFFSET_FALLBACK 2048
#endif

// In calibration function:
dc_estimate = DC_OFFSET_FALLBACK;
LOG_WRN("DC offset calibration failed, using fallback: %d", DC_OFFSET_FALLBACK);
```

---

## Summary of Required Fixes

| Issue | Priority | Location | Fix | Impact |
|-------|----------|----------|-----|--------|
| **2A: Race Condition** | 🔴 CRITICAL | Lines 749-751 | Don't reset staging buffer, use separate calib buffer OR skip reset | Prevents data corruption |
| **2B: Buffer State Loss** | 🔴 CRITICAL | Lines 749-751 | Track calibration boundaries properly | Prevents reading wrong slots |
| **3A: Calibration Timeout** | 🟠 HIGH | Line 642 | Increase timeout to 500ms, add validation | Prevents fallback to wrong DC offset |
| **4: GAIN1_4 → GAIN1_2** | 🟠 HIGH | Line 119 + 135 | Change gain, update DC offset fallback to 2048 | +10dB SNR improvement |
| **5: Slow IIR Filter** | 🟠 HIGH | Line 502 | Change >> 8 to >> 4, OR increase calibration confidence | Faster low-freq capture |
| **6: Signed Right-Shift** | 🟡 MEDIUM | Line 238 | Add unsigned cast, or add diagnostics | Prevents sign-extension corruption |
| **7: Wrong Fallback** | 🟡 MEDIUM | Line 658 | Use #define linked to actual GAIN setting | Prevents gain mismatch |

---

## Recommended Implementation Order

1. **IMMEDIATELY:** Fix Issue 2A (race condition) - this could corrupt data
2. **IMMEDIATELY:** Fix Issue 3A (timeout) - quick change, high impact
3. **NEXT:** Implement Issue 4 (GAIN1_2) - biggest SNR improvement
4. **NEXT:** Fix Issue 5 (faster IIR) - improves low-freq response
5. **OPTIONAL:** Fix Issues 6 & 7 - defensive improvements

---

## Testing Procedure After Fixes

1. **Enable logging** to see:
   - DC calibration start/completion
   - Calibration sample count vs expected
   - DC offset measured value
   - ISR events and overruns

2. **Verify with debug injection** (TEST command) first:
   - No SAADC involved, only signal processing
   - If TEST works but live recording doesn't, issue is SAADC/analog

3. **Record real audio** and check:
   - WAV file is no longer pure noise/buzz
   - Music or speech is audible
   - HR/RR accuracy hasn't degraded

4. **Monitor logs for warnings:**
   - DC calibration timeouts
   - Staging overruns
   - Buffer misalignments


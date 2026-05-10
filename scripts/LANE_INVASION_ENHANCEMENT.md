# Lane Invasion Sensor Enhancement - Legal/Illegal Classification

## Summary

Enhanced the **LaneInvasionSensor** to classify each lane crossing as **legal** or **illegal** based on:
1. Vehicle heading direction (Left vs Right)
2. Lane marking permissions (`lane_change` field)
3. Mismatch detection (illegal if crossing in restricted direction)

---

## Changes Made

### 1. **LaneInvasionSensor** (`carla_app/sensores/lane_invasion_sensor.py`)

**New Tracking Fields:**
- `legal_invasions_count`: Total legal lane crossings
- `illegal_invasions_count`: Total illegal lane crossings
- `unknown_invasions_count`: Unclassified invasions
- `last_vehicle_heading`: Previous heading for delta calculation

**New Methods:**

1. **`_determine_crossing_direction(current_heading)`**
   - Calculates heading change (yaw delta)
   - Returns: `'Left'` or `'Right'`
   - Logic: Positive yaw delta → Left turn, Negative → Right turn

2. **`_classify_invasion(crossing_direction, lane_change_str)`**
   - Classifies each invasion based on lane_change permissions
   - Returns: `'legal'`, `'illegal'`, or `'unknown'`
   - Mapping:
     - `lane_change = 'Both'` → Always legal
     - `lane_change = 'Left'` + crossing_direction = Left → legal
     - `lane_change = 'Right'` + crossing_direction = Right → legal
     - `lane_change = 'None'` → Always illegal
     - Mismatch → illegal

3. **Getter Methods:**
   - `get_legal_invasions_count()`
   - `get_illegal_invasions_count()`
   - `get_unknown_invasions_count()`
   - `get_last_invasion_type()`

**Updated Callback (`_on_invasion`):**
- Now classifies each crossed marking individually
- Updates separate counters for legal/illegal/unknown
- Stores classification in invasion history

---

### 2. **CarlaEnv** (`carla_env.py`)

**Tracking Variables (in `__init__`):**
```python
self.last_legal_invasions_count = 0
self.last_illegal_invasions_count = 0
self.last_unknown_invasions_count = 0
```

**Reset Logic (in `reset()`):**
```python
self.last_legal_invasions_count = 0
self.last_illegal_invasions_count = 0
self.last_unknown_invasions_count = 0
```

**Reward Function Enhancement (in `step()`):**
```python
# NEW: Differentiated penalties
new_legal_invasions = current_legal - last_legal
new_illegal_invasions = current_illegal - last_illegal

if new_illegal_invasions > 0:
    reward -= 8.0 * new_illegal_invasions  # Stronger penalty
    
if new_legal_invasions > 0:
    reward -= 2.0 * new_legal_invasions   # Lighter penalty
```

**Termination Condition:**
- Episode terminates if `illegal_invasions >= 5`
- Reason: `'illegal_lane_invasions'`

**Enhanced Telemetry Print:**
Added new section showing:
```
--- lane_invasions ---
legal_invasions  : X
illegal_invasions: Y
unknown_invasions: Z
```

---

## Classification Logic Examples

### Scenario 1: Legal Left Lane Change
```
lane_change: 'Both'
crossing_direction: 'Left'
Result: ✓ LEGAL (allowed in any direction)
```

### Scenario 2: Illegal Right Lane Change
```
lane_change: 'Left'       (only allow LEFT changes)
crossing_direction: 'Right'
Result: ✗ ILLEGAL (attempted RIGHT when not permitted)
```

### Scenario 3: Restricted Lane
```
lane_change: 'None'
crossing_direction: 'Left' or 'Right'
Result: ✗ ILLEGAL (no lane changes allowed)
```

---

## Reward Structure

| Event | Penalty | Notes |
|-------|---------|-------|
| **Illegal Lane Invasion** | -8.0 per invasion | Strong disincentive |
| **Legal Lane Invasion** | -2.0 per invasion | Light penalty (still not ideal) |
| **Collision** | -10.0 | Terminal event |
| **Heading >90°** | -5.0 | Terminal event |
| **Off-road** | -10.0 | Terminal event |
| **5+ Illegal Invasions** | Terminal | Episode ends |

---

## Testing

All classification logic validated with `test_invasion_classification.py`:

✓ 8/8 Classification scenarios passed
✓ 8/8 Heading delta calculations passed

Test coverage:
- Both directions allowed (Both)
- Single direction allowed (Left/Right)
- Mismatches (wrong direction)
- Restricted lanes (None)

---

## Integration with Training

The enhanced sensor now provides intelligent classification that enables:

1. **Smarter Reward Shaping**: Agent learns that some lane changes are legal/expected
2. **Better Convergence**: Reduced penalty for "allowed" lane changes helps training stability
3. **Safety Emphasis**: Strong penalty for illegal lane changes enforces traffic rules
4. **Detailed Metrics**: Separate tracking enables analysis of agent behavior

---

## Files Modified

- ✓ `carla_app/sensores/lane_invasion_sensor.py` - Full enhancement
- ✓ `carla_env.py` - Reward integration + telemetry
- ✓ `test_invasion_classification.py` - New validation test

---

## Next Steps

1. Run `test_dqn.py` with agent to see how legal/illegal invasions are triggered
2. Monitor telemetry output during training to verify classification
3. Adjust penalty weights (`-8.0` and `-2.0`) based on training convergence
4. Visualize invasion history to verify correct classification in replays

import machine
import time

# Initialize potential GPIO pins with PULL_UP (for buttons connected to GND)
pins_pullup = {}
pins_pulldown = {}

scan_gpis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28]

for gp in scan_gpis:
    try:
        # We test both pull-up and pull-down configurations
        pins_pullup[gp] = machine.Pin(gp, machine.Pin.IN, machine.Pin.PULL_UP)
    except Exception:
        pass

print("==================================================")
print("     PICO BUTTON PIN DIAGNOSTIC TOOL (PULL-UP)")
print("==================================================")
print("Wiring Assumption: Button connected between GP Pin and GND.")
print("Press your button. The target pin should toggle from 1 -> 0.")
print("Scanning... (Press Ctrl+C to stop and try PULL-DOWN mode)")
print("--------------------------------------------------")

last_states_pu = {gp: p.value() for gp, p in pins_pullup.items()}

# Print initial states
active_low_pins = [gp for gp, val in last_states_pu.items() if val == 0]
if active_low_pins:
    print(f"Note: Pins starting LOW (already grounded): GP{active_low_pins}")

while True:
    time.sleep(0.05)
    for gp, p in pins_pullup.items():
        val = p.value()
        if val != last_states_pu[gp]:
            print(f"[CHANGE] GP{gp} changed: {last_states_pu[gp]} -> {val} (Pull-Up Mode)")
            last_states_pu[gp] = val

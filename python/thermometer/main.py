import machine
import dht
import time

rtc = machine.RTC()

pin_capteur = machine.Pin(13, machine.Pin.IN, machine.Pin.PULL_UP)
capteur = dht.DHT22(pin_capteur)

# LED Configuration
blue = machine.Pin(21, machine.Pin.OUT)
green = machine.Pin(8, machine.Pin.OUT)
yellow = machine.Pin(3, machine.Pin.OUT)
red = machine.Pin(19, machine.Pin.OUT)

# Ordered list of LEDs from coldest to hottest
led_zones = [blue, green, yellow, red]

# Button configurations (PULL_DOWN to 3.3V)
pin_off = machine.Pin(18, machine.Pin.IN, machine.Pin.PULL_DOWN)
pin_on = machine.Pin(7, machine.Pin.IN, machine.Pin.PULL_DOWN)
display_enabled = True
last_off_state = 0
last_on_state = 0

temperature_precedente = None
trend = "stable"

def get_zone_index(temp):
    """Returns the index (0 to 3) of the current temperature zone"""
    if temp <= 18:
        return 0  # Blue
    elif 18 < temp <= 21:
        return 1  # Green
    elif 21 < temp <= 25:
        return 2  # Yellow
    else:
        return 3  # Red

def update_predictive_leds(temp, current_trend, blink_state):
    # 1. Turn all LEDs off first
    blue.value(0); green.value(0); yellow.value(0); red.value(0)
    
    current_idx = get_zone_index(temp)
    
    # 2. Set current temperature zone to SOLID ON
    led_zones[current_idx].value(1)
    
    # 3. Handle the trend (Blink the neighbor LED)
    if current_trend == "up":
        target_idx = current_idx + 1
        if target_idx < len(led_zones):
            led_zones[target_idx].value(blink_state)
            
    elif current_trend == "down":
        target_idx = current_idx - 1
        if target_idx >= 0:
            led_zones[target_idx].value(blink_state)

print("Démarrage du thermomètre prédictif persistant...")

blink = 0
loop_tick = 0

while True:
    time.sleep(0.05) # Snappy 50ms tick loop
    loop_tick = (loop_tick + 1) % 200
    
    # --- POLL BUTTON STATES (Every 50ms) ---
    # Check OFF Button (GP18)
    off_val = pin_off.value()
    if off_val != last_off_state:
        time.sleep(0.01) # 10ms debounce
        if pin_off.value() == off_val:
            last_off_state = off_val
            if off_val == 1: # Pressed (Transition Low -> High)
                display_enabled = False
                # Instantly turn off all LEDs
                blue.value(0); green.value(0); yellow.value(0); red.value(0)
                print("BUTTON: GP18 pressed - Display OFF")
                
    # Check ON Button (GP07)
    on_val = pin_on.value()
    if on_val != last_on_state:
        time.sleep(0.01) # 10ms debounce
        if pin_on.value() == on_val:
            last_on_state = on_val
            if on_val == 1: # Pressed (Transition Low -> High)
                display_enabled = True
                print("BUTTON: GP07 pressed - Display ON")
    
    # --- HEARTBEAT & BLINK (Every 0.5s -> 10 ticks) ---
    if loop_tick % 10 == 0:
        blink = 1 - blink
    
    # --- SENSOR SAMPLING (Every 10 seconds -> 200 ticks) ---
    if loop_tick == 0:
        try:
            capteur.measure()
            temperature = capteur.temperature()
            humidite = capteur.humidity()
            
            now = rtc.datetime()
            heures, minutes, secondes = now[4], now[5], now[6]
            
            # Evaluate true trend without resetting to stable if temp is equal
            if temperature_precedente is not None:
                if temperature > temperature_precedente:
                    trend = "up"
                elif temperature < temperature_precedente:
                    trend = "down"
            
            temperature_precedente = temperature
            
            # Console debug print
            print(f"\r{heures:02d}:{minutes:02d}:{secondes:02d} | Temp: {temperature:.1f}°C | Humidite: {humidite:.1f}% | Tendance: {trend.upper()}", end="")
            
        except OSError:
            print("\r[Erreur de lecture AM2302]", end="")
            
    # --- REFRESH LED DISPLAY (Every 0.5s blink cycle) ---
    if loop_tick % 10 == 0 and temperature_precedente is not None:
        if display_enabled:
            update_predictive_leds(temperature, trend, blink)
        else:
            blue.value(0); green.value(0); yellow.value(0); red.value(0)
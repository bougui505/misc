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

temperature_precedente = None
trend = "stable"
sensor_timer = 0 

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

while True:
    # Heartbeat loop running every 0.5s for snappy LED blinking
    time.sleep(0.5)
    blink = 1 - blink 
    
    # --- SENSOR SAMPLING (Every 10 seconds) ---
    if sensor_timer == 0:
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
                # Si la température est identique, 'trend' reste inchangé ("up" ou "down")
            
            temperature_precedente = temperature
            
            # Console debug print
            print(f"\r{heures:02d}:{minutes:02d}:{secondes:02d} | Temp: {temperature:.1f}°C | Tendance: {trend.upper()}", end="")
            
        except OSError:
            print("\r[Erreur de lecture AM2302]", end="")
            
    # --- REFRESH LED DISPLAY ---
    if temperature_precedente is not None:
        update_predictive_leds(temperature, trend, blink)
        
    sensor_timer = (sensor_timer + 1) % 20
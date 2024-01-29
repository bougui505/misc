#!/usr/bin/env python3

import time
from pigpio_dht import DHT22

gpio = 4 # BCM Numbering
sensor = DHT22(gpio)
result = sensor.sample(samples=5)
# print(result)
# {'temp_c': 18.2, 'temp_f': 64.8, 'humidity': 67.0, 'valid': True}
print(time.time(), result["temp_c"], result["humidity"])

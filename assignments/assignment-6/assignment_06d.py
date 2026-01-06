###############################################################################
# File Name: assignment_06d.py
#
# Description: This script connects to a public MQTT broker and publishes 
# randomly generated temperature (TEMPERATURERHYS) and humidity (HUMIDITYRHYS) 
# values.
#
# Record of Revisions (Date | Author | Change):
# 09/29/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import paho.mqtt.client as mqtt
from random import uniform

# Broker
broker = 'test.mosquitto.org'

# Connect to Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(broker, 1883, 60)

# Publish Temperature/Humidity Data
temp = uniform(50, 52)
client.publish("TEMPERATURERHYS", temp)
hum = uniform(60, 80)
client.publish("HUMIDITYRHYS", hum)



###############################################################################
# File Name: assignment_06c.py
#
# Description: This program connects to a public MQTT broker and publishes a 
# randomly generated rain value to the RAINRHYS topic.
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
rain = uniform(0, 2)
client.publish("RAINRHYS", rain)



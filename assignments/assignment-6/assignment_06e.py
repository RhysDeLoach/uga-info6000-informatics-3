###############################################################################
# File Name: assignment_06e.py
#
# Description: This script connects to a public MQTT broker, subscribes to the 
# topics TEMPERATURERHYS, HUMIDITYRHYS, and RAINRHYS, and collects messages 
# into a dictionary until 50 messages are received, printing each received 
# message as it stores it.
#
# Record of Revisions (Date | Author | Change):
# 09/29/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import paho.mqtt.client as mqtt
import time

# Broker
broker = 'test.mosquitto.org'

# Connect to Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Initialize Variables
messageCount = 0
data = {'TEMPERATURERHYS':[], 'HUMIDITYRHYS':[], 'RAINRHYS':[]}

# Subscribe to topics
def on_connect(client, userdata, flags, reasonCode, properties):
    print(f"Connected with reason code {reasonCode}")
    client.subscribe("TEMPERATURERHYS")
    client.subscribe("HUMIDITYRHYS")
    client.subscribe("RAINRHYS")

# Read Message
def on_message(client, userdata, msg):
    global messageCount
    messageCount += 1
    topic = msg.topic
    message = msg.payload.decode("utf-8")
    data[topic].append(message)
    print(f'Message Received: Stored {topic.lower()} reading of {message}')

client.on_connect = on_connect
client.on_message = on_message
client.connect(broker, 1883, 60)
client.loop_start()

while messageCount < 50:
    time.sleep(0.1)

client.loop_stop()
###############################################################################
# File Name: assignment_06f.py
#
# Description: This script connects to a public MQTT broker, subscribes to the 
# topics TEMPERATURERHYS and HUMIDITYRHYS, and continuously listens for 
# messages. Each received message is decoded and stored in a dictionary, with 
# the total message count incremented and printed. Messages from unknown topics 
# are flagged.
#
# Record of Revisions (Date | Author | Change):
# 09/29/2025 | Rhys DeLoach | Initial creation
###############################################################################


# Import Libraries
import paho.mqtt.client as mqtt

# Broker
broker = 'test.mosquitto.org'

# Initialize Variables
messageCount = 0
data = {'TEMPERATURERHYS': [], 'HUMIDITYRHYS': []}

# Subscribe to topics
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("TEMPERATURERHYS")
    client.subscribe("HUMIDITYRHYS")

# Read Message
def on_message(client, userdata, msg):
    global messageCount
    messageCount += 1  # Add to Message Count
    topic = msg.topic
    message = msg.payload.decode("utf-8")
    if topic in data:
        data[topic].append(message)  # Store in Dictionary
        print(f'Message Received: Stored {topic} reading of {message}')
    else:
        print(f'Unknown topic {topic} received.')

# Setup client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker, 1883, 60)
client.loop_forever()
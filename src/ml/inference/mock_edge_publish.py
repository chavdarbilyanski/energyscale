import paho.mqtt.client as mqtt
import time
import json

# --- Configuration ---
# These are the default settings for an EMQX broker.
# IMPORTANT: Note the use of port 8083, which is the default for MQTT over WebSockets.
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 8083  # Default WebSocket port for EMQX
MQTT_WEBSOCKET_PATH = "/mqtt"  # The default path

# The topic we will publish to
MQTT_TOPIC = "openems"

def on_connect(client, userdata, flags, rc, properties=None):
    """
    Callback function that is called when the client successfully connects to the broker.
    """
    if rc == 0:
        print("Successfully connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

def publish_message(client):
    """
    Constructs a sample message and publishes it to the specified topic.
    """
    # Create a sample payload. JSON is a very common format for MQTT messages.
    # We'll mimic a simplified OpenEMS data structure.
    message_payload = {
        "timestamp": int(time.time()),
        "values": {
            "ess0/Soc": 75.5,
            "ess0/ActivePower": 20000, # Positive for discharging
            "_sum/GridActivePower": 20500 
        },
        "source_id": "cloud_arbitrage_model"
    }

    # Convert the Python dictionary to a JSON string
    message_str = json.dumps(message_payload)

    # The publish() function sends the message.
    # It returns a result code; 0 means success.
    result = client.publish(MQTT_TOPIC, message_str)
    
    # Check if the message was successfully queued for sending
    if result.rc == 0:
        print(f"Sent message to topic '{MQTT_TOPIC}': {message_str}")
    else:
        print(f"Failed to send message to topic {MQTT_TOPIC}")

def main():
    """
    Main function to set up the client and publish a message.
    """
    # --- Client Initialization ---
    # This is the most critical part for WebSockets.
    # 1. We specify the transport protocol as "websockets".
    # 2. We use CallbackAPIVersion.VERSION2 for compatibility with paho-mqtt v2.x.
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")

    # 2. Set the WebSocket path.
    client.ws_set_options(path=MQTT_WEBSOCKET_PATH)

    # Assign the on_connect callback function
    client.on_connect = on_connect

    # --- Connect and Publish ---
    try:
        # Connect to the broker
        print(f"Connecting to broker at ws://{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}{MQTT_WEBSOCKET_PATH}...")
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)

        # The Paho-MQTT library's network loop runs in a background thread.
        # loop_start() is non-blocking and handles reconnections automatically.
        client.loop_start()

        # Give the client a moment to establish the connection
        time.sleep(1) 

        # Publish our message
        publish_message(client)

        # Give the client a moment to send the message before exiting
        time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop the network loop and disconnect cleanly
        client.loop_stop()
        client.disconnect()
        print("Disconnected from broker.")

if __name__ == '__main__':
    main()
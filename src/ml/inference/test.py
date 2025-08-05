# src/ml/inference/test.py (With longer delay for receive test)
import websocket
import json
import time

def on_message(ws, message):
    print("Received message:", message)
    # Profit-Max Mock: Parse bundle for RL obs
    data = json.loads(message)
    if 'Soc' in data:
        soc = data['Soc']
        pv_power = data.get('PvPower', 0)
        # Example: Build partial obs, log for oracle tuning
        print(f"Mock RL Obs: Battery %: {soc/100}, PV kWh: {pv_power}")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_reason):
    print("Connection closed")

def on_open(ws):
    # Subscribe to topic
    ws.send(json.dumps({"op": "subscribe", "topic": "openems/test_plant1/bess_state"}))
    print("Subscribed to topic")

    # Publish mock atomic bundle
    mock_data = {
        "Soc": 75.5,
        "Power": -20.0,
        "Voltage": 400.0,
        "Current": 50.0,
        "PvPower": 15.5  # For hybrid
    }
    ws.send(json.dumps({"op": "publish", "topic": "openems/test_plant1/bess_state", "message": mock_data}))
    print("Published atomic data")

    # Longer delay to allow manual publish from Dashboard or another script
    time.sleep(30)  # Run a separate publish during this time

if __name__ == "__main__":
    websocket.enableTrace(True)  # Debug
    ws = websocket.WebSocketApp("ws://localhost:8083/mqtt",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                subprotocols=['mqtt'])
    ws.run_forever()
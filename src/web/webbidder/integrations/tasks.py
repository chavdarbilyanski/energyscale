# inverter_integration/tasks.py
from celery import shared_task
import paho.mqtt.client as mqtt
from .models import InverterData  # Assume model for storage
import json, time

MQTT_URL = 'localhost'  # Local dev; prod: emqx-service-url
MQTT_PORT = 1883  # 8883 for TLS
MQTT_USER = 'admin'
MQTT_PASS = 'securepass'

@shared_task
def mqtt_subscribe(plant_id):
    def on_connect(client, userdata, flags, rc):
        client.subscribe(f"openems/{plant_id}/#")

    def on_message(client, userdata, msg):
        if msg.topic.endswith('/bess_state'):
            try:
                data = json.loads(msg.payload.decode())
                soc = data.get('Soc', 0)
                power = data.get('Power', 0)
                voltage = data.get('Voltage', 0)
                current = data.get('Current', 0)
                pv_power = data.get('PvPower', 0)  # For hybrid
                timestamp = data.get('timestamp')
                # Store atomic
                InverterData.objects.create(
                    plant_id=plant_id, soc_percent=soc, power=power, voltage=voltage, 
                    current=current, pv_power=pv_power, timestamp=timestamp
                )
                # RL Inference Trigger
                current_price = get_current_price()  # API/DB
                forecasts = run_lstm_forecast(current_price)  # lstm_forecaster.py
                obs = build_obs(current_price, soc / 100, voltage, current, pv_power, forecasts)  # Extend batteryEnv.py for atomic data
                model = mlflow.pyfunc.load_model("models:/BatteryPPOModel/Latest")
                action = model.predict(obs)[0]
                send_command(plant_id, action)  # Publish back
            except Exception as e:
                import mlflow
                mlflow.log_metric("parse_error", 1)
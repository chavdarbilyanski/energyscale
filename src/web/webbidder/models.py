from django.db import models

class TradeSimulation(models.Model):
    csv_file = models.FileField(upload_to='csvs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    battery_charge = models.FloatField(default=0.0)  # Final battery charge
    total_profit = models.FloatField(default=0.0)    # Total profit
    results = models.TextField(blank=True)          # Store detailed results as JSON or text

    def __str__(self):
        return f"Simulation {self.id} - {self.uploaded_at}"
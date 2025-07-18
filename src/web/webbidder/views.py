# src/web/webbidder/views.py

import logging
from django.shortcuts import render
from .forms import CSVUploadForm
from . import services

logger = logging.getLogger(__name__)

def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            logger.info("Form is valid. Starting processing.")
            csv_file = request.FILES['csv_file']
            
            try:
                # --- Run the RL Model Simulation ---
                rl_total_profit, rl_results = services.run_rl_model_simulation(
                    csv_file,
                    form.cleaned_data['max_battery_capacity'],
                    form.cleaned_data['charge_discharge_rate']
                )

                # --- Calculate the Perfect Score ---
                perfect_profit = services.calculate_globally_optimal_profit(
                    csv_file,
                    form.cleaned_data['max_battery_capacity'],
                    form.cleaned_data['charge_discharge_rate']
                )

                # --- Calculate Performance Score ---
                performance_score = 0
                if perfect_profit > 0:
                    performance_score = (rl_total_profit / perfect_profit) * 100

                context = {
                    'rl_total_profit': rl_total_profit,
                    'perfect_profit': perfect_profit,
                    'performance_score': performance_score,
                    'results': rl_results, # Pass the RL results for the detailed log
                }
                return render(request, 'results.html', context)

            except Exception as e:
                logger.exception("An error occurred during processing.")
                form.add_error(None, f"An unexpected error occurred: {e}")
    else:
        form = CSVUploadForm()
    
    return render(request, 'upload.html', {'form': form})
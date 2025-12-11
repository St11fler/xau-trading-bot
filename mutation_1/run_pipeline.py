#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт, който последователно стартира всички нужни етапи (collect, train short, train long, generate predictions,
train action classifier, backtest) и накрая пуска бота.
Показва прогрес и приблизително време до край.

Актуализиран така, че ако стъпката вече има артефакт (напр. 'longterm_model.keras'), да се прескача.
"""

import subprocess
import time
import sys
import os

def run_pipeline():
    # Списък от стъпки: (описание, python_file, artifact_file)
    # Ако artifact_file съществува, прескачаме стъпката.
    steps = [
        ("Събиране на данни (collect)", "collect_new.py", "XAUUSD_data_multiple_timeframes.csv"),  
        ("Обучение на short-term Transformer", "short_transformer.py", "scalping_model.keras"),
        ("Обучение на long-term Transformer", "long_transformer.py", "longterm_model.keras"),
        ("Генериране на прогнози", "generate_predictions.py", None),  # Тук няма явен artifact
        ("Обучение на LightGBM класификатор", "action_lgbm.py", "action_classifier.pkl"),
        ("Backtest на стратегията", "backtest_new.py", None),
        ("Пускане на бота (live)", "bot_new.py", None)
    ]

    total_steps = len(steps)
    completed_steps = 0

    # Съхраняваме времената за всяка стъпка, за да може да изчислим колко време приблизително остава
    times = []

    print("=== Започваме изпълнението на целия pipeline... ===\n")

    for idx, (description, script_file, artifact_file) in enumerate(steps, start=1):
        print(f"[Стъпка {idx}/{total_steps}] - {description}")

        # Ако artifact_file съществува, прескачаме стъпката
        if artifact_file and os.path.exists(artifact_file):
            print(f"Артефактът '{artifact_file}' вече съществува - прескачам стъпката.\n")
            completed_steps += 1
            continue

        step_start_time = time.time()
        print(f"Стартирам скрипта: {script_file}")

        # Стартираме скрипта със subprocess.run()
        try:
            subprocess.run([sys.executable, script_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ГРЕШКА] Скриптът {script_file} завърши с грешка (exit code = {e.returncode}). Спирам pipeline-а.")
            sys.exit(1)
        except FileNotFoundError:
            print(f"\n[ГРЕШКА] Не намирам скрипта: {script_file}. Проверете името или пътя.")
            sys.exit(1)

        # Измерваме колко е отнело
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        times.append(step_time)

        completed_steps += 1
        print(f"Завършихме стъпка '{description}' за {step_time:.1f} секунди.\n")

        # Изчисляваме приблизителен оставащ time-to-finish
        if completed_steps < total_steps:
            avg_time = sum(times) / len(times)  # Средно време за една стъпка
            steps_left = total_steps - completed_steps
            eta = steps_left * avg_time
            print(f"Прогрес: {completed_steps}/{total_steps} стъпки ({(completed_steps/total_steps)*100:.1f}%).")
            print(f"Очаквано време до край: ~{eta:.1f} секунди.\n")
        else:
            print("Всички стъпки са завършени!\n")

    print("=== Всички скриптове са изпълнени успешно или вече бяха прескочени (артефакти налични). ===")

if __name__ == "__main__":
    run_pipeline()

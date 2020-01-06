import psutil
import os
import time

import classifier

import psutil
from multiprocessing import Process

xtrainfile = os.getenv("TRAINING_DATA")
ytrainfile = os.getenv("TRAINING_LABELS")
xvalidationfile = os.getenv("VALIDATION_DATA")
yvalidationfile = os.getenv("VALIDATION_LABELS")

# run this in a separate thread


def get_resources_informations(report_id):
    memory_infos = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent()
    memory_used = memory_infos.total - memory_infos.available
    memory_used_gb = memory_used / 1024 / 1024 / 1024
    memory_used_percentage = memory_used / memory_infos.total * 100

    print("[RESOURCES] report #{}; Memory used: {:.2f} GB ({:.2f}%). CPU usage: {:.2f}%".format(report_id, memory_used_gb, memory_used_percentage, cpu_usage))

run_classifier = Process(target=classifier.classify, args=[xtrainfile, ytrainfile, xvalidationfile, yvalidationfile])

run_classifier.start()
report_id = 0
while run_classifier.is_alive():
    run_memory_monitor = Process(target=get_resources_informations, args=[report_id])
    run_memory_monitor.start()
    report_id += 1
    run_memory_monitor.join()
    time.sleep(10)

run_classifier.join()
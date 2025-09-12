import psutil
import csv
import time

filename = "system_usage.csv"

# Create CSV file with headers
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp",
        "cpu_percent",
        "per_core_percent",
        "ram_percent",
        "cpu_temperature",
        "fan_speed_rpm",
        "battery_percent",
        "battery_plugged",
        "battery_time_left_sec",
        "pid",
        "process_name",
        "process_cpu_percent",
        "process_memory_percent"
    ])

def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps: 
            return temps["coretemp"][0].current
        elif "cpu_thermal" in temps:
            return temps["cpu_thermal"][0].current
        else:
            return None
    except Exception:
        return None

def get_battery_status():
    try:
        battery = psutil.sensors_battery()
        if battery:
            return battery.percent, battery.power_plugged, battery.secsleft
        else:
            return None, None, None
    except Exception:
        return None, None, None

def get_fan_speed():
    try:
        fans = psutil.sensors_fans()
        if fans:
            for name, entries in fans.items():
                return entries[0].current
        return None
    except Exception:
        return None

def log_system_data():
    cpu_percent = psutil.cpu_percent(interval=1)
    per_core_percent = psutil.cpu_percent(interval=1, percpu=True)
    ram_percent = psutil.virtual_memory().percent

    cpu_temp = get_cpu_temperature()
    fan_speed = get_fan_speed()
    batt_percent, batt_plugged, batt_time_left = get_battery_status()

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    cpu_percent,
                    per_core_percent,
                    ram_percent,
                    cpu_temp,
                    fan_speed,
                    batt_percent,
                    batt_plugged,
                    batt_time_left,
                    proc.info['pid'],
                    proc.info['name'],
                    proc.info['cpu_percent'],
                    proc.info['memory_percent']
                ])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

# --- Run ---
if __name__ == "__main__":
    while True:
        log_system_data()
        time.sleep(5)

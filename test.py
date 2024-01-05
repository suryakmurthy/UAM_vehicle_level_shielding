import json
import matplotlib.pyplot as plt

# Load data from JSON file
with open('/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/log/training_full.json', 'r') as file:
    data = json.load(file)

# Extracting data for plotting
scenario_nums = [i for i, entry in enumerate(data)]
scenario_nums_mod = [scenario_nums[i] for i in range(0, len(scenario_nums)-1)]
shield_totals = [entry['shield_total'] for entry in data]
los_totals = [entry['los'] for entry in data]
shield_totals_i = [entry['shield_total_intersection'] for entry in data]
shield_totals_j = [entry['shield_total_route'] for entry in data]
max_travel_times = [entry['max_travel_time'] for entry in data]
max_travel_times = [max_travel_times[i] for i in range(0, len(max_travel_times)-1)]

# Plotting
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

# Plotting the data in separate subplots
plt.plot(scenario_nums, shield_totals, marker='o', linestyle='--', color='purple')
plt.xlabel('Scenario Number')
plt.ylabel('Shield Total')
plt.title('Scenario Number vs Shield Total')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, shield_totals_i, marker='o', linestyle='--', color='blue')
plt.xlabel('Scenario Number')
plt.ylabel('Shield Interventions at Intersections')
plt.title('Scenario Number vs Shield Interventions at Intersections')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, shield_totals_j, marker='o', linestyle='--', color='red')
plt.xlabel('Scenario Number')
plt.ylabel('Shield Interventions at Routes')
plt.title('Scenario Number vs Shield Interventions at Routes')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, los_totals, marker='o', linestyle='--', color='orange')
plt.xlabel('Scenario Number')
plt.ylabel('NMAC Occurrences')
plt.title('Scenario Number vs NMAC Occurrences')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums_mod, max_travel_times, marker='o', linestyle='--', color='green')
plt.xlabel('Scenario Number')
plt.ylabel('Max Travel Time')
plt.title('Scenario Number vs Max Travel Time')

plt.tight_layout()
plt.show()


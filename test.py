import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON file
with open('/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/log/eval/full_training_version_30_1_0.json', 'r') as file:
    data = json.load(file)

# Extracting data for plotting
scenario_nums = [i for i, entry in enumerate(data)]
scenario_nums_mod = [scenario_nums[i] for i in range(0, len(scenario_nums)-1)]
shield_totals = [entry['shield_total'] for entry in data]
los_totals = [entry['los'] for entry in data]
print("LOS mean: ", np.mean(los_totals))

shield_totals_i = [entry['shield_total_intersection'] for entry in data]
shield_totals_j = [entry['shield_total_route'] for entry in data]
print("shield_total_intersection mean: ", np.mean(shield_totals_i))
print("shield_total_route mean: ", np.mean(shield_totals_j))
max_travel_times = [entry['max_travel_time'] for entry in data]
print("max travel time: ", np.mean(max_travel_times))
# max_travel_times = [max_travel_times[i] for i in range(0, len(max_travel_times)-1)]
RM = 30
st_df = pd.DataFrame(shield_totals)
st_df = st_df.rolling(RM).mean()

si_df = pd.DataFrame(shield_totals_i)
si_df = si_df.rolling(RM).mean()

sr_df = pd.DataFrame(shield_totals_j)
sr_df = sr_df.rolling(RM).mean()




# plt.plot(nmac_no_tm.rolling(RM).mean())
# plt.plot(nmac_uneq.rolling(RM).mean())
# plt.plot(st_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Total Shield Interventions')
# plt.title('Scenario Number vs Total Shield Interventions (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# plt.plot(sr_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Routes')
# plt.title('Scenario Number vs Shield Interventions at Routes (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# plt.plot(si_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Intersections')
# plt.title('Scenario Number vs Shield Interventions at Intersections (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# max_val = np.argmax(shield_totals)
# del shield_totals[max_val]
# del scenario_nums[max_val]
# del los_totals[max_val]
# del shield_totals_i[max_val]
# del shield_totals_j[max_val]
# del max_travel_times[max_val]
# del scenario_nums_mod[max_val]

# Plotting
plt.figure(figsize=(30, 8))  # Adjust the figure size as needed

# Plotting the data in separate subplots
plt.plot(scenario_nums, shield_totals, marker='o', linestyle='--', color='purple')
plt.xlabel('Scenario Number')
plt.ylabel('Total Shield Interventions')
plt.title('Scenario Number vs Total Shield Interventions (Reward = -0.0001)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, shield_totals_i, marker='o', linestyle='--', color='blue')
plt.xlabel('Scenario Number')
plt.ylabel('Shield Interventions at Intersections')
plt.title('Scenario Number vs Shield Interventions at Intersections (Reward = -0.0001)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, shield_totals_j, marker='o', linestyle='--', color='red')
plt.xlabel('Scenario Number')
plt.ylabel('Shield Interventions at Routes')
plt.title('Scenario Number vs Shield Interventions at Routes (Reward = -0.0001)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, los_totals, marker='o', linestyle='--', color='orange')
plt.xlabel('Scenario Number')
plt.ylabel('NMAC Occurrences')
plt.title('Scenario Number vs NMAC Occurrences (Reward = -0.0001)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, max_travel_times, marker='o', linestyle='--', color='green')
plt.xlabel('Scenario Number')
plt.ylabel('Max Travel Time')
plt.title('Scenario Number vs Max Travel Time (Reward = -0.0001)')

plt.tight_layout()
plt.show()


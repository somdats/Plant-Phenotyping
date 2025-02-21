import numpy as np


# Load data from file
data = np.loadtxt('D:/4d_plant_registration_data/tomato/tomato_values_comp_p2p.txt')

# Calculate mean
avg = np.mean(data)

# Calculate min
min_value = np.min(data)

# Calculate max
max_value = np.max(data)

# Calculate standard deviation
std_dev = np.std(data)

print("average:", avg)
print("Standard Deviation:", std_dev)
print("min:", min_value)
print("max_value:", max_value)

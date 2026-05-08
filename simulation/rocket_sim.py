import matplotlib.pyplot as plt
import numpy as np
import math

# Fundamental Simulation Variables
# You can tweak these parameters to match your specific rocket motor and mass
mass = 1.23  # kg
gravity = 9.802  # m/s^2
initial_thrust = 132.0  # N (Peak thrust at ignition)
initial_thrust_duration = 0.2  # s (How long the initial peak lasts)
normal_thrust = 46.0  # N (Sustained thrust)
burn_time = 1.6  # s (Total motor burn time)

# Aerodynamic parameters
area = 0.00785  # m^2
Cd = 0.75
rho = 1.225  # kg/m^3

# Launch parameters
launch_angle_deg = 85.0  # Degrees from horizontal. 90 = straight up.
launch_angle_rad = math.radians(launch_angle_deg)

# Initial conditions
x = 0.0  # Horizontal displacement (m)
y = 0.0  # Vertical displacement (m)
v_x = 0.0  # Horizontal velocity (m/s)
v_y = 0.0  # Vertical velocity (m/s)

# Simulation parameters
dt = 0.01  # Time step (s)
time_elapsed = 0.0

# Data recording for visualization
times = []
x_vals = []
y_vals = []
v_y_vals = []
v_x_vals = []
accel_y_vals = []

print("Starting simulation...")

# Run simulation until the rocket hits the ground (y < 0 after launch)
while time_elapsed == 0 or y >= 0:
    # Determine current thrust
    if time_elapsed < initial_thrust_duration:
        thrust = initial_thrust
    elif time_elapsed < burn_time:
        thrust = normal_thrust
    else:
        thrust = 0.0

    # Calculate aerodynamic drag
    velocity_mag = math.sqrt(v_x**2 + v_y**2)
    drag_force = 0.5 * rho * Cd * area * (velocity_mag**2)

    if velocity_mag > 0:
        drag_x = drag_force * (v_x / velocity_mag)
        drag_y = drag_force * (v_y / velocity_mag)
    else:
        drag_x = 0.0
        drag_y = 0.0

    # Calculate forces
    thrust_x = thrust * math.cos(launch_angle_rad)
    thrust_y = thrust * math.sin(launch_angle_rad)
    
    force_net_x = thrust_x - drag_x
    force_net_y = thrust_y - (mass * gravity) - drag_y

    # Calculate acceleration
    a_x = force_net_x / mass
    a_y = force_net_y / mass

    # Update velocity
    v_x += a_x * dt
    v_y += a_y * dt

    # Update displacement
    x += v_x * dt
    y += v_y * dt

    # Record data
    times.append(time_elapsed)
    x_vals.append(x)
    y_vals.append(y)
    v_x_vals.append(v_x)
    v_y_vals.append(v_y)
    accel_y_vals.append(a_y)

    # Increment time
    time_elapsed += dt
    
    # Safety breakout if it goes too long (e.g. into orbit or hovering)
    if time_elapsed > 1000:
        print("Simulation timeout.")
        break

print(f"Simulation complete.")
print(f"Apogee (Max Height): {max(y_vals):.2f} m")
print(f"Total Time of Flight: {time_elapsed:.2f} s")
print(f"Max Horizontal Distance: {max(x_vals):.2f} m")

# Visualization with Matplotlib
plt.figure(figsize=(12, 8))

# 1. Trajectory (X vs Y)
plt.subplot(2, 2, 1)
plt.plot(x_vals, y_vals, 'b-', label='Trajectory')
plt.title('Rocket Trajectory')
plt.xlabel('Horizontal Displacement (m)')
plt.ylabel('Vertical Displacement (m)')
plt.grid(True)
plt.legend()

# 2. Vertical Altitude vs Time
plt.subplot(2, 2, 2)
plt.plot(times, y_vals, 'g-')
plt.title('Altitude vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Displacement (m)')
plt.grid(True)

# 3. Velocities vs Time
plt.subplot(2, 2, 3)
plt.plot(times, v_y_vals, 'r-', label='Vertical Velocity')
plt.plot(times, v_x_vals, 'c-', label='Horizontal Velocity')
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

# 4. Vertical Acceleration vs Time
plt.subplot(2, 2, 4)
plt.plot(times, accel_y_vals, 'm-', label='Vertical Acceleration')
plt.axvline(x=burn_time, color='k', linestyle='--', label='Motor Burnout')
plt.title('Vertical Acceleration vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

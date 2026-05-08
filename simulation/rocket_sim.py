import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import math

# 1. New Core Concepts & Variables
mass = 1.23  # kg
gravity = 9.802  # m/s^2

# Thrust Profile (4x Estes D12 cluster)
initial_thrust = 132.0  # N (Peak thrust at ignition)
initial_thrust_duration = 0.2  # s (How long the initial peak lasts)
normal_thrust = 46.0  # N (Sustained thrust)
burn_time = 1.6  # s (Total motor burn time)

# Aerodynamic & Physical parameters
area = 0.00785  # m^2
Cd = 0.75
rho = 1.225  # kg/m^3
height = 0.99  # m
diameter = 0.1  # m

# Moment Arms (Distance from base)
cg_from_base = 0.35  # m
cp_from_base = 0.60  # m
moment_arm_tvc = cg_from_base  # Base to CG
moment_arm_drag = cp_from_base - cg_from_base  # CP to CG

# Moment of Inertia (uniform cylinder approximation)
inertia = (1/12) * mass * (height**2) + (1/4) * mass * ((diameter/2)**2)

# 2. The Controller Section
MAX_GIMBAL_ANGLE_DEG = 5.0
MAX_GIMBAL_ANGLE_RAD = math.radians(MAX_GIMBAL_ANGLE_DEG)

# PD Controller Gains
Kp_gain = 1.5
Kd_gain = 0.25

# Launch parameters
launch_angle_deg = 85.0  # Degrees from horizontal
launch_angle_rad = math.radians(launch_angle_deg)

# Initial conditions (Translational)
x = 0.0  # Horizontal displacement (m)
y = 0.0  # Vertical displacement (m)
v_x = 0.0  # Horizontal velocity (m/s)
v_y = 0.0  # Vertical velocity (m/s)

# 3. Add Rotational State Variables
pitch_angle = 0.0  # radians (0 means aligned with launch angle)
pitch_rate = 0.0   # radians/s
gimbal_angle = 0.0 # radians

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
pitch_vals = []  # Array for pitch data
gimbal_vals = [] # Array for gimbal data

print("Starting 2D Rigid Body Dynamics simulation...")

# Run simulation until the rocket hits the ground (y < 0 after launch)
while time_elapsed == 0 or y >= 0:
    # Determine current thrust magnitude
    if time_elapsed < initial_thrust_duration:
        thrust = initial_thrust
    elif time_elapsed < burn_time:
        thrust = normal_thrust
    else:
        thrust = 0.0

    # 4. Update the Physics Loop
    
    # Run Controller
    target_gimbal = (Kp_gain * pitch_angle) + (Kd_gain * pitch_rate)
    
    # Actuation Lag/Limit
    target_gimbal = np.clip(target_gimbal, -MAX_GIMBAL_ANGLE_RAD, MAX_GIMBAL_ANGLE_RAD)
    gimbal_angle += (target_gimbal - gimbal_angle) * 0.25

    # Global orientation
    body_angle = launch_angle_rad + pitch_angle
    
    # Total Thrust Vector
    thrust_dir = body_angle + gimbal_angle
    thrust_x = thrust * math.cos(thrust_dir)
    thrust_y = thrust * math.sin(thrust_dir)
    
    # Thrust Radial component
    thrust_radial = thrust * math.sin(gimbal_angle)
    
    # Drag Vector
    velocity_mag = math.sqrt(v_x**2 + v_y**2)
    drag_force = 0.5 * rho * Cd * area * (velocity_mag**2)

    if velocity_mag > 0:
        drag_x = drag_force * (v_x / velocity_mag)
        drag_y = drag_force * (v_y / velocity_mag)
        
        # Calculate Drag Radial component
        v_angle = math.atan2(v_y, v_x)
        aoa = body_angle - v_angle
        drag_radial = drag_force * math.sin(aoa)
    else:
        drag_x = 0.0
        drag_y = 0.0
        drag_radial = 0.0

    # Calculate Torques
    torque_tvc = thrust_radial * moment_arm_tvc
    torque_drag = drag_radial * moment_arm_drag
    net_torque = torque_tvc - torque_drag

    # Calculate Rotational Acceleration
    angular_acceleration = net_torque / inertia

    # Update Rotational State
    pitch_rate += angular_acceleration * dt
    pitch_angle += pitch_rate * dt

    # Update Translational State
    force_net_x = thrust_x - drag_x
    force_net_y = thrust_y - (mass * gravity) - drag_y

    a_x = force_net_x / mass
    a_y = force_net_y / mass

    v_x += a_x * dt
    v_y += a_y * dt

    x += v_x * dt
    y += v_y * dt

    # Record data
    times.append(time_elapsed)
    x_vals.append(x)
    y_vals.append(y)
    v_x_vals.append(v_x)
    v_y_vals.append(v_y)
    accel_y_vals.append(a_y)
    pitch_vals.append(math.degrees(pitch_angle))
    gimbal_vals.append(math.degrees(gimbal_angle))

    # Increment time
    time_elapsed += dt
    
    # Safety breakout if it goes too long
    if time_elapsed > 1000:
        print("Simulation timeout.")
        break

print(f"Simulation complete.")
print(f"Apogee (Max Height): {max(y_vals):.2f} m")
print(f"Total Time of Flight: {time_elapsed:.2f} s")
print(f"Max Horizontal Distance: {max(x_vals):.2f} m")

# 5. Update Visualization (Animation)
fig, ax = plt.subplots(figsize=(6, 10))

# Define Artists
rocket_body = patches.Rectangle((-diameter/2, 0), diameter, height, fc='blue')
thrust_vector, = ax.plot([], [], '-', color='red', linewidth=2)
trajectory_path, = ax.plot([], [], 'g-', alpha=0.5)
telemetry_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ground_line, = ax.plot([-100, 100], [0, 0], 'k-', linewidth=2)

def init():
    ax.add_patch(rocket_body)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 100)
    return rocket_body, thrust_vector, trajectory_path, telemetry_text, ground_line

def update(frame):
    t = times[frame]
    cx = x_vals[frame]
    cy = y_vals[frame]
    vx = v_x_vals[frame]
    vy = v_y_vals[frame]
    pitch = pitch_vals[frame]  # degrees
    gimbal = gimbal_vals[frame]  # degrees
    
    # Calculate Thrust status
    if t < initial_thrust_duration:
        thrust_mag = initial_thrust
    elif t < burn_time:
        thrust_mag = normal_thrust
    else:
        thrust_mag = 0.0

    body_angle_deg = launch_angle_deg + pitch
    rot_angle_deg = body_angle_deg - 90.0

    tr = transforms.Affine2D().rotate_deg_around(0, 0, rot_angle_deg) + transforms.Affine2D().translate(cx, cy) + ax.transData
    
    rocket_body.set_xy((-diameter/2, -cg_from_base))
    rocket_body.set_width(diameter)
    rocket_body.set_height(height)
    rocket_body.set_transform(tr)

    trajectory_path.set_data(x_vals[:frame+1], y_vals[:frame+1])

    if thrust_mag > 0:
        thrust_len = thrust_mag / 50.0  # Scale down for visual
        flame_angle_rad = math.radians(body_angle_deg + gimbal) + math.pi
        flame_end_x = cx - cg_from_base * math.cos(math.radians(body_angle_deg)) + thrust_len * math.cos(flame_angle_rad)
        flame_end_y = cy - cg_from_base * math.sin(math.radians(body_angle_deg)) + thrust_len * math.sin(flame_angle_rad)
        base_x = cx - cg_from_base * math.cos(math.radians(body_angle_deg))
        base_y = cy - cg_from_base * math.sin(math.radians(body_angle_deg))
        
        thrust_vector.set_data([base_x, flame_end_x], [base_y, flame_end_y])
        thrust_vector.set_linestyle('-')
        thrust_vector.set_color('red')
    else:
        base_x = cx - cg_from_base * math.cos(math.radians(body_angle_deg))
        base_y = cy - cg_from_base * math.sin(math.radians(body_angle_deg))
        flame_angle_rad = math.radians(body_angle_deg + gimbal) + math.pi
        flame_end_x = base_x + 0.5 * math.cos(flame_angle_rad)
        flame_end_y = base_y + 0.5 * math.sin(flame_angle_rad)
        thrust_vector.set_data([base_x, flame_end_x], [base_y, flame_end_y])
        thrust_vector.set_linestyle('--')
        thrust_vector.set_color('gray')

    telemetry_text.set_text(
        f"Time: {t:.2f} s\n"
        f"Altitude: {cy:.2f} m\n"
        f"Velocity: {math.sqrt(vx**2+vy**2):.2f} m/s\n"
        f"Pitch: {pitch:.2f}°\n"
        f"Gimbal: {gimbal:.2f}°"
    )

    window_height = 40
    window_width = 20
    ax.set_xlim(cx - window_width/2, cx + window_width/2)
    min_y = max(-5, cy - 10)
    ax.set_ylim(min_y, min_y + window_height)

    return rocket_body, thrust_vector, trajectory_path, telemetry_text, ground_line

ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=20)
ani.save('simulation/rocket_launch.mp4', writer='ffmpeg', fps=50)

print("Animation logic added. The simulation will now generate 'simulation/rocket_launch.mp4' after running.")

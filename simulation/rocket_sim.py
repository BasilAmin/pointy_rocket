import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import math

# 1. Physical Parameters
initial_total_mass = 1.23  # kg
gravity = 9.802  # m/s^2

# Dynamic Mass Parameters
initial_propellant_mass = 0.098  # kg (approx 24.5g per D12 motor x 4)
dry_mass = initial_total_mass - initial_propellant_mass

# Center of Gravity Parameters
cg_prop_from_base = 0.05  # m
# We must find the dry CG such that the initial total CG is 0.35m
# Total_CG = (Dry_Mass * CG_dry + Prop_Mass * CG_prop) / Total_Mass
# 0.35 = (1.132 * CG_dry + 0.098 * 0.05) / 1.23
# 0.4305 = 1.132 * CG_dry + 0.0049
# CG_dry = 0.4256 / 1.132 = 0.37597
cg_dry_from_base = 0.37597  # m

# Thrust Profile (4x Estes D12 cluster)
# Using realistic D12 thrust curve points multiplied by 4
thrust_curve_time = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.8, 1.2, 1.5, 1.6, 1.7])
thrust_curve_force = np.array([100.0, 132.0, 132.0, 90.0, 48.0, 44.0, 42.0, 40.0, 35.0, 0.0, 0.0])
burn_time = 1.6  # s

# Calculate Cumulative Impulse for Mass flow rate
cum_impulse = np.zeros_like(thrust_curve_force)
for i in range(1, len(thrust_curve_time)):
    dt = thrust_curve_time[i] - thrust_curve_time[i-1]
    avg_f = 0.5 * (thrust_curve_force[i] + thrust_curve_force[i-1])
    cum_impulse[i] = cum_impulse[i-1] + avg_f * dt
total_impulse = cum_impulse[-1]

# Aerodynamic & Physical parameters
area_base = 0.00785  
Cd_base = 0.75
rho = 1.225  
height = 0.99  
diameter = 0.1  

# Parachute specs
parachute_area = 0.28  # m^2
parachute_cd = 1.5

# Fixed Aerodynamic Center of Pressure
cp_from_base = 0.60  # m

# 2. Control Parameters
MAX_GIMBAL_ANGLE_DEG = 8.0
MAX_GIMBAL_ANGLE_RAD = math.radians(MAX_GIMBAL_ANGLE_DEG)
SERVO_TAU = 0.05  

Kp_gain = 5.0
Kd_gain = 1.2
Ki_gain = 0.5

# 3. Kalman Filter Class
class KalmanFilter1D:
    def __init__(self, initial_state):
        self.x = np.array(initial_state, dtype=float)
        self.P = np.eye(2)
        self.Q = np.array([[1e-5, 0.0], 
                           [0.0, 1e-3]])
        self.H = np.eye(2)

    def predict(self, dt):
        A = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z, R):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x

# 4. Simulation & Initial Conditions
dt = 0.01  
launch_angle_deg = 88.0
launch_angle_rad = math.radians(launch_angle_deg)

X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def get_physics_properties(t):
    """Calculates instantaneous mass, cg, inertia, and thrust."""
    if t >= thrust_curve_time[-1]:
        thrust = 0.0
        current_impulse = total_impulse
    else:
        thrust = np.interp(t, thrust_curve_time, thrust_curve_force)
        current_impulse = np.interp(t, thrust_curve_time, cum_impulse)
        
    prop_mass = initial_propellant_mass * max(0.0, 1.0 - (current_impulse / total_impulse))
    mass = dry_mass + prop_mass
    
    cg_from_base = ((dry_mass * cg_dry_from_base) + (prop_mass * cg_prop_from_base)) / mass
    inertia = (1/12) * mass * (height**2) + (1/4) * mass * ((diameter/2)**2)
    
    return thrust, mass, cg_from_base, inertia

# Environmental Parameters
surface_wind_x = 2.0  # m/s
wind_shear_gradient = 0.05  # m/s per meter of altitude

def get_wind(y):
    return surface_wind_x + wind_shear_gradient * max(0.0, y)

def get_derivatives(t, state, est_pitch, est_pitch_rate, est_pitch_int):
    x, y, vx, vy, pitch, pitch_rate, gimbal = state
    
    thrust, mass, cg_from_base, inertia = get_physics_properties(t)
    
    moment_arm_tvc = cg_from_base
    moment_arm_drag = cp_from_base - cg_from_base
        
    target_gimbal = -((Kp_gain * est_pitch) + (Kd_gain * est_pitch_rate) + (Ki_gain * est_pitch_int))
    target_gimbal = np.clip(target_gimbal, -MAX_GIMBAL_ANGLE_RAD, MAX_GIMBAL_ANGLE_RAD)
    
    dgimbal_dt = (target_gimbal - gimbal) / SERVO_TAU
    
    body_angle = launch_angle_rad + pitch
    
    # Parachute deployment logic
    if vy < 0 and t > burn_time:
        area = parachute_area
        Cd = parachute_cd
    else:
        area = area_base
        Cd = Cd_base

    # Wind and Relative Velocity
    v_wind_x = get_wind(y)
    v_rel_x = vx - v_wind_x
    v_rel_y = vy  # Assuming negligible vertical wind

    velocity_mag = math.sqrt(v_rel_x**2 + v_rel_y**2)
    drag_force = 0.5 * rho * Cd * area * (velocity_mag**2)
    
    if velocity_mag > 1e-6:
        drag_x = drag_force * (v_rel_x / velocity_mag)
        drag_y = drag_force * (v_rel_y / velocity_mag)
        
        v_angle = math.atan2(v_rel_y, v_rel_x)
        aoa = body_angle - v_angle
        drag_radial = drag_force * math.sin(aoa)
    else:
        drag_x = 0.0
        drag_y = 0.0
        drag_radial = 0.0
        
    thrust_dir = body_angle + gimbal
    thrust_x = thrust * math.cos(thrust_dir)
    thrust_y = thrust * math.sin(thrust_dir)
    
    force_net_x = thrust_x - drag_x
    force_net_y = thrust_y - (mass * gravity) - drag_y
    
    ax = force_net_x / mass
    ay = force_net_y / mass
    
    thrust_radial = thrust * math.sin(gimbal)
    
    torque_tvc = thrust_radial * moment_arm_tvc
    torque_drag = drag_radial * moment_arm_drag
    net_torque = torque_tvc - torque_drag
    
    angular_accel = net_torque / inertia
    
    return np.array([vx, vy, ax, ay, pitch_rate, angular_accel, dgimbal_dt])

def rk4_step(t, state, dt, est_pitch, est_pitch_rate, est_pitch_int):
    k1 = get_derivatives(t, state, est_pitch, est_pitch_rate, est_pitch_int)
    k2 = get_derivatives(t + dt/2, state + (dt/2) * k1, est_pitch, est_pitch_rate, est_pitch_int)
    k3 = get_derivatives(t + dt/2, state + (dt/2) * k2, est_pitch, est_pitch_rate, est_pitch_int)
    k4 = get_derivatives(t + dt, state + dt * k3, est_pitch, est_pitch_rate, est_pitch_int)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

times = []
x_vals, y_vals = [], []
vx_vals, vy_vals = [], []
vmag_vals = []
ax_vals, ay_vals = [], []
amag_vals = []
pitch_vals, gimbal_vals = [], []

est_pitch_vals, est_pitch_rate_vals = [], []
meas_pitch_vals, meas_pitch_rate_vals = [], []
true_pitch_rate_vals = []
mass_vals = []
thrust_vals = []

time_elapsed = 0.0
kf = KalmanFilter1D(initial_state=[0.0, 0.0])
est_pitch = 0.0
est_pitch_rate = 0.0
est_pitch_int = 0.0

print("Starting RK4 Rigid Body Dynamics + IMU + Dynamic Mass + Wind Simulation...")

while time_elapsed == 0 or X[1] >= 0:
    thrust, mass, cg_from_base, inertia = get_physics_properties(time_elapsed)
    derivs = get_derivatives(time_elapsed, X, est_pitch, est_pitch_rate, est_pitch_int)
    
    ax_true = derivs[2]
    ay_true = derivs[3]
    pitch_true = X[4]
    pitch_rate_true = X[5]

    times.append(time_elapsed)
    x_vals.append(X[0])
    y_vals.append(X[1])
    vx_vals.append(X[2])
    vy_vals.append(X[3])
    vmag_vals.append(math.sqrt(X[2]**2 + X[3]**2))
    ax_vals.append(ax_true)
    ay_vals.append(ay_true)
    amag_vals.append(math.sqrt(ax_true**2 + ay_true**2))
    pitch_vals.append(math.degrees(pitch_true))
    gimbal_vals.append(math.degrees(X[6]))
    true_pitch_rate_vals.append(math.degrees(pitch_rate_true))
    mass_vals.append(mass)
    thrust_vals.append(thrust)

    A_global_x = ax_true
    A_global_y = ay_true + gravity
    body_angle_true = launch_angle_rad + pitch_true
    
    A_long = A_global_x * math.cos(body_angle_true) + A_global_y * math.sin(body_angle_true)
    A_lat = -A_global_x * math.sin(body_angle_true) + A_global_y * math.cos(body_angle_true)
    
    A_long_noisy = A_long + np.random.normal(0, 0.5)
    A_lat_noisy = A_lat + np.random.normal(0, 0.5)
    
    measured_body_angle = math.atan2(A_long_noisy, A_lat_noisy)
    measured_pitch = measured_body_angle - launch_angle_rad
    
    while measured_pitch - est_pitch > math.pi: measured_pitch -= 2*math.pi
    while measured_pitch - est_pitch < -math.pi: measured_pitch += 2*math.pi
        
    measured_pitch_rate = pitch_rate_true + np.random.normal(0, math.radians(2.0)) + math.radians(0.5)
    
    meas_pitch_vals.append(math.degrees(measured_pitch))
    meas_pitch_rate_vals.append(math.degrees(measured_pitch_rate))

    kf.predict(dt)
    
    a_mag = math.sqrt(A_long_noisy**2 + A_lat_noisy**2)
    g_error = abs(a_mag - gravity)
    R_pitch = 1e-2 + (g_error * 0.1)**2 
    R_rate = (math.radians(2.0))**2 
    
    R = np.array([[R_pitch, 0], 
                  [0, R_rate]])
    
    est_state = kf.update(np.array([measured_pitch, measured_pitch_rate]), R)
    est_pitch = est_state[0]
    est_pitch_rate = est_state[1]
    
    est_pitch_vals.append(math.degrees(est_pitch))
    est_pitch_rate_vals.append(math.degrees(est_pitch_rate))
    
    # Integral update
    est_pitch_int += est_pitch * dt
    # Anti-windup
    est_pitch_int = np.clip(est_pitch_int, -math.radians(20), math.radians(20))
    
    X = rk4_step(time_elapsed, X, dt, est_pitch, est_pitch_rate, est_pitch_int)
    time_elapsed += dt
    
    if time_elapsed > 1000:
        print("Simulation timeout.")
        break

print(f"Simulation complete.")
print(f"Apogee (Max Height): {max(y_vals):.2f} m")
print(f"Total Time of Flight: {time_elapsed:.2f} s")
print(f"Max Horizontal Distance: {max(x_vals):.2f} m")
print(f"Final Dry Mass: {mass_vals[-1]:.3f} kg")

# Static Plots
fig_static = plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.plot(x_vals, y_vals, 'b-', label='Trajectory')
plt.title('Rocket Trajectory')
plt.xlabel('Horizontal Displacement (m)')
plt.ylabel('Vertical Displacement (m)')
plt.grid(True)

plt.subplot(2, 4, 2)
plt.plot(times, y_vals, 'g-')
plt.title('Altitude vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Displacement (m)')
plt.grid(True)

plt.subplot(2, 4, 3)
plt.plot(times, vy_vals, 'r-', label='Vertical Velocity')
plt.plot(times, vx_vals, 'c-', label='Horizontal Velocity')
plt.plot(times, vmag_vals, 'k:', label='Total Velocity')
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(times, mass_vals, 'm-', linewidth=2)
plt.title('Mass vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.grid(True)

plt.subplot(2, 4, 5)
plt.plot(times, pitch_vals, 'k-', linewidth=2, label='True Pitch')
plt.plot(times, meas_pitch_vals, 'r.', alpha=0.2, markersize=2, label='Noisy Accel Pitch')
plt.plot(times, est_pitch_vals, 'g-', linewidth=1.5, label='Filtered Pitch')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5)
plt.title('Pitch Angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle (degrees)')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')

plt.subplot(2, 4, 6)
plt.plot(times, true_pitch_rate_vals, 'k-', linewidth=2, label='True Rate')
plt.plot(times, meas_pitch_rate_vals, 'r.', alpha=0.2, markersize=2, label='Noisy Gyro Rate')
plt.plot(times, est_pitch_rate_vals, 'g-', linewidth=1.5, label='Filtered Rate')
plt.title('Pitch Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Rate (deg/s)')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')

plt.subplot(2, 4, 7)
plt.plot(times, thrust_vals, 'orange', linewidth=2)
plt.title('Thrust vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.grid(True)

plt.subplot(2, 4, 8)
plt.plot(times, ay_vals, 'm-', label='Vertical Accel')
plt.plot(times, ax_vals, 'c-', label='Horizontal Accel')
plt.plot(times, amag_vals, 'k:', label='Total Accel')
plt.axvline(x=burn_time, color='k', linestyle='--', label='Motor Burnout')
plt.title('Acceleration vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()

plt.tight_layout()
fig_static.savefig('D:/pointy_rocket/simulation/simulation_results.png')
plt.close(fig_static)

# Animation
fig_anim, ax = plt.subplots(figsize=(6, 10))
rocket_body = patches.Rectangle((-diameter/2, 0), diameter, height, fc='blue')
thrust_vector, = ax.plot([], [], '-', color='red', linewidth=2)
trajectory_path, = ax.plot([], [], 'g-', alpha=0.5)

canopy = patches.Arc((0, 0), 2.0, 1.5, theta1=0.0, theta2=180.0, color='orange', linewidth=2)
chute_line, = ax.plot([], [], 'k-', linewidth=1)

telemetry_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ground_line, = ax.plot([-100, 100], [0, 0], 'k-', linewidth=2)

def init():
    ax.add_patch(rocket_body)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 100)
    return rocket_body, thrust_vector, trajectory_path, telemetry_text, ground_line, canopy, chute_line

def update(frame):
    t = times[frame]
    cx, cy = x_vals[frame], y_vals[frame]
    vx, vy = vx_vals[frame], vy_vals[frame]
    pitch, gimbal = pitch_vals[frame], gimbal_vals[frame]
    
    # Recalculate CG for drawing
    thrust_mag, mass_val, current_cg, _ = get_physics_properties(t)

    body_angle_deg = launch_angle_deg + pitch
    rot_angle_deg = body_angle_deg - 90.0

    tr = transforms.Affine2D().rotate_deg_around(0, 0, rot_angle_deg) + transforms.Affine2D().translate(cx, cy) + ax.transData
    rocket_body.set_xy((-diameter/2, -current_cg))
    rocket_body.set_transform(tr)

    trajectory_path.set_data(x_vals[:frame+1], y_vals[:frame+1])

    base_x = cx - current_cg * math.cos(math.radians(body_angle_deg))
    base_y = cy - current_cg * math.sin(math.radians(body_angle_deg))
    
    if thrust_mag > 0:
        thrust_len = thrust_mag / 50.0
        flame_angle_rad = math.radians(body_angle_deg + gimbal) + math.pi
        flame_end_x = base_x + thrust_len * math.cos(flame_angle_rad)
        flame_end_y = base_y + thrust_len * math.sin(flame_angle_rad)
        thrust_vector.set_data([base_x, flame_end_x], [base_y, flame_end_y])
        thrust_vector.set_linestyle('-')
        thrust_vector.set_color('red')
    else:
        flame_angle_rad = math.radians(body_angle_deg + gimbal) + math.pi
        flame_end_x = base_x + 0.5 * math.cos(flame_angle_rad)
        flame_end_y = base_y + 0.5 * math.sin(flame_angle_rad)
        thrust_vector.set_data([base_x, flame_end_x], [base_y, flame_end_y])
        thrust_vector.set_linestyle('--')
        thrust_vector.set_color('gray')

    chute_status = "Deployed" if vy < 0 and t > burn_time else "Packed"

    if vy < 0 and t > burn_time:
        if canopy not in ax.patches:
            ax.add_patch(canopy)
        top_x = cx + (height - current_cg) * math.cos(math.radians(body_angle_deg))
        top_y = cy + (height - current_cg) * math.sin(math.radians(body_angle_deg))
        canopy_x = top_x
        canopy_y = top_y + 3.0
        canopy.center = (canopy_x, canopy_y)
        chute_line.set_data([top_x, canopy_x], [top_y, canopy_y])
    else:
        if canopy in ax.patches:
            canopy.remove()
        chute_line.set_data([], [])

    telemetry_text.set_text(f"Time: {t:.2f} s\nAlt: {cy:.2f} m\nVel: {math.sqrt(vx**2+vy**2):.2f} m/s\nPitch: {pitch:.2f}°\nGimbal: {gimbal:.2f}°\nChute: {chute_status}\nMass: {mass_val:.3f} kg")
    
    window_height, window_width = 40, 20
    ax.set_xlim(cx - window_width/2, cx + window_width/2)
    min_y = max(-5, cy - 10)
    ax.set_ylim(min_y, min_y + window_height)
    return rocket_body, thrust_vector, trajectory_path, telemetry_text, ground_line, canopy, chute_line

try:
    ani = FuncAnimation(fig_anim, update, frames=len(times), init_func=init, blit=False, interval=20)
    ani.save('D:/pointy_rocket/simulation/rocket_launch.mp4', writer='ffmpeg', fps=50)
    print("Animation saved.")
except Exception as e:
    print(f"Animation skip: {e}")
finally:
    plt.close(fig_anim)

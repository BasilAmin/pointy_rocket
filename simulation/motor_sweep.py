import numpy as np
import math

print("Running Motor Parameter Sweep...")

for motor_count in [4, 3, 2, 1]:
    # Airframe mass + (0.0422 kg per motor)
    mass = 1.061 + (0.0422 * motor_count)
    
    # Thrust profile
    initial_thrust = 33.0 * motor_count
    initial_thrust_duration = 0.2
    normal_thrust = 11.5 * motor_count
    burn_time = 1.6

    gravity = 9.802

    # Aerodynamic & Physical parameters
    area = 0.00785
    Cd = 0.75
    rho = 1.225
    height = 0.99
    diameter = 0.1

    # Parachute specs
    parachute_area = 0.28
    parachute_cd = 1.5
    parachute_deployed = False

    # Moment Arms
    cg_from_base = 0.35
    cp_from_base = 0.60
    moment_arm_tvc = cg_from_base
    moment_arm_drag = cp_from_base - cg_from_base

    # Moment of Inertia
    inertia = (1/12) * mass * (height**2) + (1/4) * mass * ((diameter/2)**2)

    # Controller Section
    MAX_GIMBAL_ANGLE_RAD = math.radians(8.0)
    Kp_gain = 5.0
    Kd_gain = 1.2

    # Launch parameters
    launch_angle_deg = 88.0
    launch_angle_rad = math.radians(launch_angle_deg)

    # Initial conditions
    x = 0.0
    y = 0.0
    v_x = 0.0
    v_y = 0.0

    pitch_angle = 0.0
    pitch_rate = 0.0
    gimbal_angle = 0.0

    dt = 0.01
    time_elapsed = 0.0
    
    max_y = 0.0

    while time_elapsed == 0 or y >= 0:
        if time_elapsed < initial_thrust_duration:
            thrust = initial_thrust
        elif time_elapsed < burn_time:
            thrust = normal_thrust
        else:
            thrust = 0.0

        # Run Controller
        target_gimbal = -((Kp_gain * pitch_angle) + (Kd_gain * pitch_rate))
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
        if time_elapsed > burn_time and v_y < 0 and not parachute_deployed:
            parachute_deployed = True
            area = parachute_area
            Cd = parachute_cd

        velocity_mag = math.sqrt(v_x**2 + v_y**2)
        drag_force = 0.5 * rho * Cd * area * (velocity_mag**2)

        if velocity_mag > 0:
            drag_x = drag_force * (v_x / velocity_mag)
            drag_y = drag_force * (v_y / velocity_mag)
            
            v_angle = math.atan2(v_y, v_x)
            aoa = body_angle - v_angle
            drag_radial = drag_force * math.sin(aoa)
        else:
            drag_x = 0.0
            drag_y = 0.0
            drag_radial = 0.0

        if not parachute_deployed:
            torque_tvc = thrust_radial * moment_arm_tvc
            torque_drag = drag_radial * moment_arm_drag
            net_torque = torque_tvc - torque_drag
        else:
            target_pitch = math.radians(90.0 - launch_angle_deg)
            pitch_rate = pitch_rate * 0.85
            pitch_angle += (target_pitch - pitch_angle) * 0.15
            net_torque = 0.0

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
        
        if y > max_y:
            max_y = y

        # Increment time
        time_elapsed += dt
        
        # Safety breakout
        if time_elapsed > 1000:
            break

    print(f"Motors: {motor_count} | Launch Mass: {mass:.3f} kg | Apogee: {max_y:.1f} m | Total Time: {time_elapsed:.1f} s")

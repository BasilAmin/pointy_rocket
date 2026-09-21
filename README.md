# 🚀 Pointy Rocket V1

**Pointy Rocket** is an actively stabilised model-scale rocket developed by the Engineering Society of Castleknock College (SVCC). It uses **Thrust Vector Control (TVC)** — the same technique used in orbital-class rockets — to actively stabilise the vehicle in flight.

The project spans custom PCB hardware design, embedded flight firmware, and a high-fidelity physics simulation, all living in this repository.

---

> **An SVCC Project with support from RS Components**
> <img width="100" height="100" alt="RS Logo" src="https://github.com/BasilAmin/pointy_rocket/blob/main/Media/svcclogo.png" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="100" height="100" alt="RS Logo" src="https://github.com/BasilAmin/pointy_rocket/blob/main/Media/RSlogo.png" />

---

## Authors

- James Harcourt
- Basil Amin
- Ashkan Samali
- Feargal Browne
- Ben Liu
- Ryan Doran

---

## Project Overview

| Component     | Details                                                             |
|---------------|---------------------------------------------------------------------|
| Stabilisation | 2-axis Thrust Vector Control (TVC) via servo-gimballed motor mount  |
| Motors        | 4× Estes D12 (clustered)                                            |
| Microcontroller | Teensy 4.1 (NXP i.MX RT1062, Cortex-M7 @ 600 MHz)               |
| IMU           | MPU6050 (6-axis accelerometer + gyroscope, I2C)                     |
| Altimeter     | MPL3115A2 barometric altimeter                                      |
| Data Logging  | SD card (SPI)                                                       |
| Recovery      | Parachute deployment                                                |

---

## Repository Structure

```
pointy_rocket/
├── Dictator_Flight_Firmware/   # Main embedded flight firmware (Teensy 4.1 / PlatformIO)
├── Firmware_test/              # Firmware testing scripts and utilities
├── hardware/                   # KiCad PCB schematic & layout
│   └── pcb/
│       └── Pointy R1 Flight controller/
│           ├── *.kicad_sch
│           ├── *.kicad_pcb
│           ├── symbols.kicad_sym
│           └── footprints.pretty/
├── simulation/                 # Python flight simulation suite
│   ├── rocket_sim.py           # Full 6-DOF 3D simulation (RK4 + Quaternions)
│   ├── tvc_sim.py              # TVC-controlled flight engine
│   ├── rocketpy_engine.py      # RocketPy aerodynamics integration
│   ├── rocketpy_adapter.py     # Hybrid engine orchestrator
│   ├── motor_curve.py          # Thrust curve parser (.eng / .rse / .csv)
│   ├── motor_sweep.py          # Motor configuration parameter sweep
│   ├── gui_server.py           # Browser-based TVC simulator GUI
│   ├── plots.py                # Standalone matplotlib figure generator
│   ├── pygame_simulation.py    # Pygame visualisation
│   ├── requirements.txt
│   └── simulation_results.png
├── Media/                      # Logos and media assets
├── Pointy R1 Flight controller.csv  # BOM / component list
├── sim_specs.txt               # Simulation variable reference
└── .gitignore
```

---

## Firmware

The flight computer runs on a **Teensy 4.1**, programmed via PlatformIO with the Arduino framework. The firmware reads IMU data, runs a PID controller, and drives the two TVC servos in real time.

### Hardware

| Component       | Details                                          |
|-----------------|--------------------------------------------------|
| Microcontroller | Teensy 4.1 (NXP i.MX RT1062, Cortex-M7 @ 600 MHz) |
| IMU             | MPU6050 (6-axis accelerometer + gyroscope, I2C)  |
| Altimeter       | MPL3115A2 barometric altimeter                   |
| Actuation       | 2× Servo (dual-axis TVC gimbal)                  |
| Data Logging    | SD card (SPI)                                    |
| Power           | AMASS XT60 connector                             |
| Ignition        | N-MOSFET (D2PAK) motor ignition circuit          |

### Dependencies

| Library                   | Version  |
|---------------------------|----------|
| `arduino-libraries/Servo` | `^1.3.0` |
| `electroniccats/MPU6050`  | `^1.4.4` |
| `arduino-libraries/SD`    | `^1.3.0` |

### Build & Flash

**Prerequisites:** [PlatformIO](https://platformio.org/) (CLI or VS Code extension)

```bash
# Clone the repo
git clone https://github.com/BasilAmin/pointy_rocket.git
cd pointy_rocket/Dictator_Flight_Firmware

# Build
pio run

# Flash to Teensy (press the button on the board if needed)
pio run --target upload

# Open serial monitor
pio device monitor
```

---

## Hardware / PCB

The custom flight controller PCB is designed in **KiCad** and lives in `hardware/pcb/Pointy R1 Flight controller/`. Custom symbols and footprints are included for:

- Teensy 4.1 (form-factor reference)
- MPU6050 breakout (GY-521 module)
- MPL3115A2 barometric altimeter
- AMASS XT60 power connector
- N-MOSFET (D2PAK) motor ignition circuit
- Tactile and slide switches
- Passive components (diode, SOT-23 transistor)

To open the design, install [KiCad 7+](https://www.kicad.org/) and open `Pointy R1 Flight controller.kicad_pro`.

---

## Simulation

The simulation suite models the full flight from ignition through parachute landing. It uses a hybrid approach combining the trusted aerodynamics of **RocketPy** with a custom in-house **TVC 6-DOF engine**.

### Architecture

Because Pointy Rocket is a finless TVC rocket (unstable by design), a purely passive simulation cannot fly it. The two engines are combined to play to each other's strengths:

- **RocketPy** (`rocketpy_engine.py`) provides trusted aerodynamics: Barrowman centre of pressure, static margin, a real atmosphere model. It cannot model thrust vectoring, so it cannot fly the rocket passively.
- **The in-house TVC sim** (`tvc_sim.py`) flies the actively steered rocket (quaternion 6-DOF, RK4, PID gimbal), using RocketPy's centre of pressure and drag values instead of a rough analytic estimate.

`rocketpy_adapter.py` orchestrates both and returns side-by-side results.

### Setup

```bash
cd simulation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `numpy`, `matplotlib`, `rocketpy` (which pulls in `scipy` and `netCDF4`).

### TVC Simulator GUI

A browser-based GUI wraps both flight engines and lets you upload or fetch motor files, import body geometry, and view results interactively.

```bash
cd simulation
python gui_server.py
```

Open `http://127.0.0.1:8765`. Each run renders a **matplotlib** figure (altitude, thrust curve, TVC gimbal angles, speed/mass) shown below the live chart with a PNG download link.

The same figure can be produced from the command line:

```bash
python plots.py path/to/motor.eng    # writes matplotlib_results.png
```

### Flight computer in the loop

The TVC sim does not steer with perfect knowledge — it runs a realistic flight-computer model (`flight_computer.py`) matched to the actual hardware (MPU6050 read at its ±2 g / ±250 °/s defaults; two servos on the Arduino `Servo` library = 1° write resolution):

- **MPU6050 sensor model** — per-flight gyro/accel bias, white noise, range clipping and 16-bit quantisation.
- **Attitude estimator** — gyro integration with on-pad calibration. (In flight a rocket's accelerometer can't sense gravity — under thrust it reads thrust, in coast it's in free fall — so the accelerometer is used only on the pad to set the initial attitude and the gyro bias. This is the reference algorithm for the Teensy firmware.)
- **Discrete PID** at the configured control rate with control-loop **latency**.
- **Servo model** — slew-rate limit, command resolution, deadband and lag.

Two findings this exposes: residual **gyro-bias drift** is the dominant attitude error, and because thrust vectoring needs thrust, the rocket is only controllable **during the burn** — after burnout a finless vehicle tumbles (expected; recovery is by parachute). Stability is therefore judged over the powered phase.

### Monte-Carlo dispersion

`montecarlo.py` runs many flights with the uncertain parameters perturbed (mass, propellant, motor thrust, drag, centre of pressure, wind, thrust misalignment, sensor bias/noise) and reports the distribution of apogee, powered-phase tilt, gimbal usage and landing radius, plus the fraction of flights that stay controlled during burn. RocketPy aerodynamics are evaluated once for the nominal vehicle; flights run in parallel across CPU cores.

```bash
python montecarlo.py path/to/motor.eng 200 path/to/rocket.ork
```

### 6-DOF Flight Simulation

`rocket_sim.py` implements a full **6 Degrees of Freedom** rigid-body simulation:

- **RK4 numerical integration** for accurate state propagation
- **Quaternion-based attitude representation** (no gimbal lock)
- **PID controller** for dual-axis TVC stabilisation
- **Dynamic mass model** — propellant mass decreases with cumulative impulse
- **High-fidelity D12 thrust curve** (4× scaled cluster, from verified RASP data)
- **Aerodynamic drag** with wind shear model
- **Automatic parachute deployment** on descent

#### PID Controller Parameters

| Gain             | Value |
|------------------|-------|
| Kp               | 0.8   |
| Kd               | 1.5   |
| Ki               | 0     |
| Max gimbal angle | ±8°   |

```bash
python rocket_sim.py
```

Outputs a 4-panel plot (`simulation_results.png`) showing:

- 3D flight trajectory
- Altitude vs. time
- Dual-axis TVC gimbal angles
- Dynamic mass and thrust profile

### Motor Configuration Sweep

`motor_sweep.py` sweeps across 1–4 motor configurations to compare predicted performance across different cluster sizes.

```bash
python motor_sweep.py
```

---

## Language Breakdown

| Language   | Share  |
|------------|--------|
| C++        | 81.2%  |
| Python     | 9.7%   |
| C          | 6.0%   |
| JavaScript | 1.2%   |
| Processing | 0.9%   |
| HTML       | 0.6%   |
| Other      | 0.4%   |

---

## License

MIT License — © 2026 Basil Amin. See [LICENSE](firmware/LICENSE) for details.
## License

MIT License — © 2026 Basil Amin. See [LICENSE](firmware/LICENSE) for details.

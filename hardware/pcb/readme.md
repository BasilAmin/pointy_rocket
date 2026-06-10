# Pointy R1 Flight Controller PCB

Custom flight controller board designed in KiCad for the Pointy R1 rocket. Handles motor control, sensor fusion, audio/visual feedback, and power regulation.

## Directory Structure

```
Pointy R1 Flight controller/
├── Pointy R1 Flight controller.kicad_pro   # KiCad project file
├── Pointy R1 Flight controller.kicad_sch   # Schematic
├── Pointy R1 Flight controller.kicad_pcb   # PCB layout
├── Pointy R1 Flight controller.csv         # Bill of materials
├── 3d Models/                              # STEP files for 3D viewer
└── Pointy R1 Flight controller-backups/    # KiCad auto-backups
```

## Bill of Materials

| Ref | Qty | Part | Description |
|-----|-----|------|-------------|
| IC1 | 1 | Teensy 4.1 | Main microcontroller |
| U2 | 1 | GY-521 (MPU-6050) | 6-axis IMU (accel + gyro) |
| U3 | 1 | MPL3115A2 | Barometric pressure / altitude sensor |
| U1 | 1 | L7805CV | 5V linear voltage regulator (TO-220) |
| Q1, Q2, Q3 | 3 | IRLZ44NSTRLPBF | N-channel MOSFET for motor/actuator control |
| D1 | 1 | 1N5822 | Schottky diode (flyback protection) |
| D2, D3, D4 | 3 | 1N4007 | Rectifier diodes |
| M1, M2 | 2 | Servo header | 3-pin 2.54 mm servo connectors |
| P1–P4 | 4 | 282837-2 | 2-pos screw terminals (TE Connectivity) |
| J3 | 1 | XT60-M | Main battery connector |
| LED1 | 1 | COM-16347 | RGB LED (SparkFun) |
| LS1 | 1 | CMI-1295-0585T | Piezo buzzer |
| S2 | 1 | TS02-66-70-BK-100-LCR-D | Tactile push button |
| C1 | 1 | 10 µF | Bulk decoupling (0805) |
| C2 | 1 | 1 µF | Mid-frequency decoupling (0805) |
| C3 | 1 | 100 nF | High-frequency decoupling (0805) |
| R2, R3, R5, R7 | 4 | 470 Ω | Current-limiting resistors (0805) |
| R8 | 1 | 10 kΩ | Pull-up/pull-down resistor (0805) |
| R9 | 1 | 4.7 kΩ | Pull-up/pull-down resistor (0805) |

## Design Notes

- **Power input:** XT60 connector → L7805CV regulates down to 5 V for logic. Teensy 4.1 runs at 3.3 V internally.
- **Motor/actuator outputs:** Three IRLZ44N logic-level MOSFETs switch inductive loads; flyback diodes (1N4007) on each gate, 1N5822 on the rail.
- **IMU:** GY-521 module (MPU-6050) connected via I2C for attitude estimation.
- **Altimeter:** MPL3115A2 on I2C for altitude and vertical speed.
- **Feedback:** RGB LED and piezo buzzer for status/alarm indication.
- **Programming:** Teensy 4.1 programs over USB; no additional programmer needed.
- **PCB has passed DFM review** — Gerbers ready for manufacture (see git history).

## Opening the Project

Requires **KiCad 7+**. Open `Pointy R1 Flight controller/Pointy R1 Flight controller.kicad_pro`.

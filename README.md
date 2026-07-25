# Franka Controller

The robot infra used to collect the dataset is released as part of the [serl](https://github.com/rail-berkeley/serl/tree/main/serl_robot_infra) project and works for both Franka Emika Panda and the newer Franka Research 3 arms. 

## Structure
All robot code is structured as follows:
- `custom_server`: hosts a Flask server which sends commands to the robot via ROS.
- `franka_env`: gym env for the robot which communicates with the Flask server via POST requests.

### Prerequisites
- ROS Noetic
- Franka Panda or Franka Research 3 arm and gripper
- `libfranka>=0.8.0` and `franka_ros>=0.8.0` installed according to [Franka FCI Documentation](https://frankaemika.github.io/docs/installation_linux.html)

### Installation
```bash
cd robot_infra
conda create -n franka_controller python=3.10
conda activate franka_controller
pip install -e .
```

Disable the `franka_ros` realtime kernel constraint in `catkin_ws/src/franka_ros/franka_control/config/franka_control_node.yaml` by setting the line:
```yaml
realtime_config: ignore
```

### Usage

1. **Launch the robot server**
   The robot server runs the robot controller and a Flask server which streams robot commands to the gym environment using HTTP requests.
    ```bash
    conda activate franka_controller
    python robot_infra/custom_server.py --robot_ip 172.16.0.2 
    ```

    | Flags | Description |
    | --- | --- |
    | `--robot_ip` | IP of the robot for launching the controller |
    | `--gripper_dist` | Distance the gripper should open to. 0.09 for single-object task, 0.075 for the multi-object task |
    | `--force_base_frame` | Whether to read the end-effector force/torque information in the base frame. |

    This starts the ROS impedance controller and the HTTP server. You can test compliance by gently pushing the end effector.

    #### HTTP Server API
    The HTTP server communicates between the ROS controller and gym environments:

    | Request | Description |
    | --- | --- |
    | `startimp` | Start the impedance controller |
    | `stopimp` | Stop the impedance controller |
    | `pose` | Command robot to desired end-effector pose in base frame (xyz+quaternion) |
    | `getpos` | Return current end-effector pose (xyz+rpy) |
    | `getvel` | Return current end-effector velocity |
    | `getforce` | Return estimated force on end-effector |
    | `gettorque` | Return estimated torque on end-effector |
    | `getq` | Return current joint position |
    | `getdq` | Return current joint velocity |
    | `getjacobian` | Return current zero-jacobian |
    | `getstate` | Return all robot states |
    | `jointreset` | Perform joint reset |
    | `get_gripper` | Return current gripper position |
    | `close_gripper` | Close the gripper completely |
    | `open_gripper` | Open the gripper completely |
    | `clearerr` | Clear errors |
    | `precision_mode` | Set impedance parameters to precision mode for resets |
    | `compliance_mode` | Set impedance parameters to compliance mode for execution |
    | `telemetry/status` | GET-only diagnostic status for UDP, source freshness, runtime payload, frames, and robot errors |

    #### Quick Terminal Commands
    ```bash
    curl -X POST http://127.0.0.1:5000/activate_gripper # Activate gripper
    curl -X POST http://127.0.0.1:5000/close_gripper    # Close gripper
    curl -X POST http://127.0.0.1:5000/open_gripper     # Open gripper
    curl -X POST http://127.0.0.1:5000/getpos           # Print current EE pose
    curl -X POST http://127.0.0.1:5000/jointreset       # Perform joint reset
    curl -X POST http://127.0.0.1:5000/stopimp          # Stop impedance controller
    curl -X POST http://127.0.0.1:5000/startimp         # Start impedance controller
    ```

    #### Franka State Telemetry

    The ROS/Flask hardware server can publish the current `FrankaState` and
    zero Jacobian as a fixed 1,069-byte binary UDP packet. The UDP stream on
    port 5010 is observation-only. It does not replace the trajectory gateway
    on TCP port 4999, which continues to relay motion commands.

    ```bash
    python robot_infra/franka_server.py \
      --robot_ip 172.16.0.2 \
      --telemetry_host 20.42.0.54 \
      --telemetry_port 5010 \
      --telemetry_hz 200

    curl -s http://127.0.0.1:5000/telemetry/status
    ```

    Packets are suppressed until both state and Jacobian callbacks are ready,
    and whenever either source is older than five telemetry periods. Leaving
    `--telemetry_host` empty preserves the existing server behavior without
    opening a UDP socket or starting a telemetry thread.

    The telemetry flags above belong to `robot_infra/franka_server.py`:

    | Flag | Description |
    | --- | --- |
    | `--telemetry_host` | ROSAIC PC IPv4 destination. Empty disables UDP telemetry. |
    | `--telemetry_port` | ROSAIC UDP telemetry port. Default: 5010. |
    | `--telemetry_hz` | Franka state telemetry rate from 1 to 200 Hz. Default: 200. |

2. **Launch Gym Environment**
    Create an instance of the gym environment in a second terminal:
    ```bash
    python franka_controller.py --gripper close --port 4999
    ```

3. **Data Collection (Space Mouse)**
    Use a 3D space mouse for robot action data sampling:
    ```bash
    python data_record.py
    ```

    *Note: For low-level SpaceMouse testing, you can also use:*
    ```bash 
    python robot_infra/spacemouse/spacemouse_custom.py
    ```

4. **Kinect Recording (Optional)**
    To record high-quality Azure Kinect video data:
    ```bash
    python record_azure.py
    ```

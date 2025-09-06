import time

import mujoco
import mujoco.viewer

import numpy as np

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
INTEGRATION_DT: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
DAMPING: float = 1e-4

# Whether to enable gravity compensation.
GRAVITY_COMPENSATION: bool = True

# Simulation timestep in seconds.
DT: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
MAX_ANGVEL = 0.0


def main():
    """Begin cassie bipedal control."""

    # Load the model and data
    model = mujoco.MJModel.from_xml_path("agility_cassie/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep
    model.opt.timestep = DT

    # Name of the bodies we wish to apply gravity compenation towards
    body_names = [

    ]
    body_ids = [model.body(name).id for name in body_names]
    if GRAVITY_COMPENSATION:
        model.body_gravcomp[body_ids] = 1.0

    # Get the DoF and actuator IDS for the joints we wish to control
    joint_names = [
        # Left leg
        "left-hip-roll",  # hinge -> actuated
        "left-hip-yaw",  # hinge -> actuated
        "left-hip-pitch",  # hinge -> actuated
        "left-achilles-rod"  # ball
        "left-knee",  # hinge -> actuated
        "left-shin",  # hinge
        "left-tarsus",  # hinge
        "left-heel-spring",  # hinge
        "left-foot-crank",  # hinge
        "left-plantar-rod",  # hinge
        "left-foot",  # hinge -> actuated
        # Right leg
        "right-hip-roll",  # hinge -> actuated
        "right-hip-yaw",  # hinge -> actuated
        "right-hip-pitch",  # hinge -> actuated
        "right-achilles-rod"  # ball
        "right-knee",  # hinge -> actuated
        "right-shin",  # hinge
        "right-tarsus",  # hinge
        "right-heel-spring",  # hinge
        "right-foot-crank",  # hinge
        "right-plantar-rod",  # hinge
        "right-foot",  # hinge -> actuated
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    
    # Actuator names are not all the joint names in this case
    actuator_names = [
        "left-hip-roll",
        "left-hip-yaw",
        "left-hip-pitch",
        "left-knee",
        "left-foot",
        "right-hip-roll",
        "right-hip-yaw",
        "right-hip-pitch",
        "right-knee",
        "right-foot",
    ]
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # Initial joint configuration saved as a keyframe in the XML file
    key_id = model.key("home").id

    # Mocap body we will control with our mouse
    # mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
         # Reset the simulation to the initial keyframe.
         mujoco.mj_resetDataKeyframe(model, data, key_id)

         # Initialize the camera view to that of the free camera.
         # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

         # Toggle site frame visualization.
         viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

         while viewer.is_running():
             pass
         

if __name__ == "__main__":
    main()

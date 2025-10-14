import torch
import torch.nn.functional as F

def is_unit_quaternion(q):
    """Check if the quaternion is a unit quaternion."""
    norm = torch.norm(q)
    return torch.isclose(norm, torch.tensor(1.0, device=q.device))

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device)

def quaternion_rotate(q, p):
    """Rotate a 3D vector p using the quaternion q."""
    q_conj = quaternion_conjugate(q)
    p_q = torch.cat([torch.tensor([0.0], device=q.device), p], dim=0)  # Convert vector p to a pure quaternion (0, x, y, z)
    rotated_p = quaternion_multiply(quaternion_multiply(q, p_q), q_conj)
    return rotated_p[1:]  # Extract the rotated vector (x, y, z)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.tensor([w, x, y, z], device=q1.device)

# Given data
P_world = torch.tensor([-2.045, 1.695, 0.660], device='cuda:0')
P_robot = torch.tensor([-0.567, 1.440, 0.547], device='cuda:0')
q_robot = torch.tensor([ -0.085,0.005, -0.004, -0.996, ], device='cuda:0')
q_robot = q_robot / q_robot.norm(dim=-1, keepdim=True).clamp_min(1e-12)
is_unit = is_unit_quaternion(q_robot)
print("is unit ", is_unit)
# Step 1: Translate the point into the robot's coordinate system
P_relative = P_world - P_robot

# Step 2: Rotate the point using the robot's quaternion
P_robot_coord = quaternion_rotate(q_robot, P_relative)

print("Transformed point in robot's coordinate system:", P_robot_coord)

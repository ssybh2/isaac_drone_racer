import numpy as np

from perception.rigid_transform import RigidTransform


def test_named_transform_composition_and_inverse():
    T_ab = RigidTransform(np.eye(3), [1.0, 0.0, 0.0], to_frame="A", from_frame="B")
    T_bc = RigidTransform(np.eye(3), [0.0, 2.0, 0.0], to_frame="B", from_frame="C")
    T_ac = T_ab @ T_bc

    assert T_ac.to_frame == "A"
    assert T_ac.from_frame == "C"
    assert np.allclose(T_ac.t, [1.0, 2.0, 0.0])
    assert np.allclose((T_ac @ T_ac.inverse()).as_matrix(), np.eye(4), atol=1e-9)

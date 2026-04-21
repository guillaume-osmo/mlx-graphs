import mlx.core as mx

from mlx_graphs.chem.ode_block import ODEBlock


def test_ode_block_euler_and_rk4_smoke():
    x = mx.random.uniform(0, 1, (5, 6))
    for solver in ("euler", "rk4"):
        block = ODEBlock(input_dim=6, n_steps=3, dt=0.1, solver=solver)
        out = block(x)
        assert out.shape == (5, 3)
        assert mx.all(mx.isfinite(out)).item()

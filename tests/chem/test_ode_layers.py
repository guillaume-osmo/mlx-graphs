import mlx.core as mx

from mlx_graphs.chem.ode_layers import (
    ODE_CNN,
    ODE_GRU,
    ODE_MLP,
    ODESkipLayer,
    num_parameters,
    ode_skip_multistep,
    ode_skip_step,
)


def test_ode_skip_helpers():
    x = mx.ones((2, 3))

    def velocity(h):
        return 2.0 * h

    step = ode_skip_step(x, velocity, dt=0.25)
    multi = ode_skip_multistep(x, velocity, dt=0.25, n_steps=2)

    assert step.shape == x.shape
    assert multi.shape == x.shape
    assert mx.all(mx.isfinite(step)).item()
    assert mx.all(mx.isfinite(multi)).item()


def test_ode_skip_layer_and_ode_mlp_smoke():
    x = mx.random.uniform(0, 1, (4, 6))
    layer = ODESkipLayer(dim=6, n_steps=3, dt=0.25)
    mlp = ODE_MLP(input_dim=6, hidden_dim=8, output_dim=1, n_steps=3, dt=0.25)

    out_layer = layer(x)
    out_mlp = mlp(x, training=True)

    assert out_layer.shape == (4, 6)
    assert out_mlp.shape == (4, 1)
    assert num_parameters(mlp) > 0


def test_ode_gru_and_cnn_smoke():
    gru = ODE_GRU(input_dim=10, hidden_dim=8, n_steps=3, dt=0.25)
    cnn = ODE_CNN(in_channels=10, out_channels=8, kernel_size=3, n_steps=3, dt=0.25)

    gru_out = gru(mx.random.uniform(0, 1, (3, 6, 10)))
    cnn_out = cnn(mx.random.uniform(0, 1, (3, 8, 10)))

    assert gru_out.shape == (3, 8)
    assert cnn_out.shape == (3, 8)
    assert mx.all(mx.isfinite(gru_out)).item()
    assert mx.all(mx.isfinite(cnn_out)).item()

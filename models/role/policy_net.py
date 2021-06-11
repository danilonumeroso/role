import haiku as hk
from jax.nn import relu


def policy_fn(x):
    net = hk.nets.MLP(
        output_sizes=[2048, 1024, 512],
        activation=relu
    )

    policy_head = hk.Linear(4096)
    value_head = hk.Linear(1)

    net_out = net(x)
    return policy_head(net_out), value_head(net_out)

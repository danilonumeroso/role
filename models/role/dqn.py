import haiku as hk
import optax
import jax
import jax.numpy as np
from jax.nn import relu
from functools import partial


def policy_fn(x):
    net = hk.nets.MLP(
        output_sizes=[1024, 512, 128],
        activation=relu
    )

    q_head = hk.Linear(1)

    h = net(x)
    return q_head(h)


def train_fns(network,
              opt,
              batch_size,
              discount):

    loss_fn = partial(_loss_fn, network, discount)
    optimize = partial(_optimize,
                       opt,
                       batch_size,
                       loss_fn)

    return loss_fn, optimize


def _loss_fn(network, discount, w_policy, w_target, x):
    s, r, s_next, is_terminal = x
    q_value = network.apply(w_policy, s)
    q_target = network.apply(w_target, s_next)

    q_target = r * (1 - is_terminal) * discount * q_target

    return q_value.mean() + q_target.mean()


def _optimize(opt,
              batch_size,
              loss_fn,
              experience_replay,
              w_policy,
              w_target,
              opt_state):

    experience = experience_replay.sample(batch_size)

    states = np.stack([S for S, *_ in experience])
    next_states = [S for *_, S, _ in experience]
    rewards = np.stack([
        R for _, R, *_ in experience
    ])
    is_terminal = np.array([
        T for *_, T in experience
    ])

    loss, grads = jax.value_and_grad(loss_fn)(w_policy, w_target, (states, rewards, next_states, is_terminal))

    updates, new_opt_state = opt.update(grads, opt_state, w_policy)
    new_params = optax.apply_updates(w_policy, updates)

    return new_params, new_opt_state, loss


def update_target_net(q_params, t_params, alpha=0.99):
    return optax.incremental_update(q_params, t_params, alpha)


def eps_greedy(rng, network, params, x, eps):
    is_random_action = jax.random.uniform(next(rng)) < eps

    if is_random_action:
        range_ = np.array(range(x.shape[0]))
        return jax.random.choice(next(rng), range_), is_random_action
    else:
        return network.apply(params, x).argmax(), is_random_action

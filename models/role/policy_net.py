# import haiku as hk
# from jax.nn import relu


# def policy_fn(x):
#     net = hk.nets.MLP(
#         output_sizes=[2048, 1024, 512],
#         activation=relu
#     )

#     policy_head = hk.Linear(4096)
#     value_head = hk.Linear(1)

#     net_out = net(x)
#     return policy_head(net_out), value_head(net_out)


# def loss_fn(params, s, a, r):
#     logits, _ = network.apply(params, s)
#     log_prob = jax.nn.log_softmax(logits)
#     log_prob = jax.numpy.take_along_axis(log_prob,
#                                              a.reshape(log_prob.shape[0], 1), -1)

#     loss = (log_prob * r.sum()).sum()
#     decay = sum(p.sum() for p in jax.tree_leaves(params))
#     return loss + 1e-2 * decay


# def train_step(params, opt_state, trajectories):

#     loss = []
#     for t in trajectories:
#         s = np.array(list(map(lambda t_: t_.state, t)))
#         a = np.array(list(map(lambda t_: t_.action, t)))
#         r = np.array(list(map(lambda t_: t_.reward, t)))

#         l, grads = jax.value_and_grad(loss_fn)(params, s, a, r)
#         loss.append(l)

#         updates, new_opt_state = tx.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)

#     return new_params, new_opt_state, sum(loss) / len(loss)

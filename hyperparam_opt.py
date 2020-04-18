from ax.service.managed_loop import optimize

from main import main


parameters=[
  dict(name='trace_decay', type='range', bounds=[0.9, 0.99]),
  dict(name='ppo_clip', type='range', bounds=[0.05, 0.3]),
  dict(name='ppo_epochs', type='range', bounds=[2, 5]),
  dict(name='value_loss_coeff', type='range', bounds=[0.5, 2.0]),
  dict(name='entropy_loss_coeff', type='range', bounds=[1e-10, 1e-3], log_scale=True),
  dict(name='learning_rate', type='range', bounds=[1e-4, 5e-3], log_scale=True),
  dict(name='batch_size', type='range', bounds=[512, 4096]),
  dict(name='max_grad_norm', type='range', bounds=[0.0, 10.0])
]
best_parameters, values, experiment, model = optimize(parameters, main, minimize=False, random_seed=1)
print(best_parameters)

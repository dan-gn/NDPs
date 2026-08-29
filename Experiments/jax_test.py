import jax
import jax.numpy as jnp

print(jax.devices())

x = jnp.ones((32, 32))
y = x @ x

print(y)
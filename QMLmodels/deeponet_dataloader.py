import jax
import jax.numpy as jnp


def deeponet_dataloader(branch, trunk, target, batchsize:int=10, epoch:int=0):
    """Dataloader for DeepONet training. Uses functional paradigm like the rest of JAX.
    
    Usage: ```for a, b, c in data_loader:```
    
    Args:
        branch: branch training data of shape (samples, discretized points)
        trunk: trunk training data of shape (samples, query locations)
        target: target training data of shape (samples, targets @ query locations)
        batchsize: batch size
        epoch: used to seed JAX PRNG. pass the number of the training epoch

    Yields:
        out: Tuple of branch, trunk, and target of shapes (batch size, discretized points), (query locations, ...), (batch size, query location targets)
    """
    key = jax.random.PRNGKey(epoch)
    assert branch.shape[0] == target.shape[0]
    NS = branch.shape[0]
    MAXITR = NS // batchsize
    sarr = jax.random.permutation(key, jnp.arange(NS))
    for itr in range(MAXITR):
        group2yield = jnp.array(range(itr * batchsize, itr * batchsize + batchsize))
        ysarr = sarr[group2yield]
        yield(
            branch[ysarr],
            trunk,
            target[ysarr]
        )

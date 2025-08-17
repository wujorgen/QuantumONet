print("importing python packages")
# import pennylane as qml
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os
import sys
import json

from copy import deepcopy
from tqdm import tqdm, trange

from deeponet_dataloader import deeponet_dataloader


if len(sys.argv) >= 2:
    NGRID = int(sys.argv[1])
    print("Input detected for number of grid points.")
else:
    print("No input for number of grid points detected. Defaulting...")
    NGRID = 11  # 9, 11, 13, 15


class DeepONet1D(eqx.Module):
    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    bias: jax.Array

    def __init__(self, in_size_branch, trunk_width, branch_width, trunk_depth, branch_depth, interact_size, activation, *, key):
        """
        For simplicity, branch and trunk nets are identical MLP here.
        These could be CNNs or anything else in practice.
        """
        b_key, t_key = jr.split(key)
        self.branch_net = eqx.nn.MLP(
            in_size=in_size_branch,
            out_size=interact_size,
            width_size=branch_width,
            depth=branch_depth,
            activation=activation,
            key=b_key
        )
        self.trunk_net = eqx.nn.MLP(
            in_size=1,
            out_size=interact_size,
            width_size=trunk_width,
            depth=trunk_depth,
            activation=activation,
            # normally MLPs dont have activation @ final output layer
            # but the trunk net gets it here
            # final_activation=activation,
            key=t_key
        )
        self.bias = jnp.zeros((1,))

    def __call__(self, x_branch, x_trunk):
        """Forward pass. 
        
        Note the split branch and trunk inputs, 
        unlike PyTorch or TensorFlow which take a tuple.
        
        x_branch.shape = (in_size_branch,)
        x_trunk.shape = (1,)
        
        return should just be a scalar value
        """
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        # perform element-wise multiplication
        inner_product = jnp.sum(branch_out * trunk_out, keepdims=True)
        inner_product += self.bias
        return inner_product[0]


def rmse_error(pred, ref):
    mse = jnp.mean((pred - ref)**2)
    return jnp.sqrt(mse)


@eqx.filter_value_and_grad  # changes function signature -> returns loss value and gradient
def loss_fn(model, branch_in, trunk_in, target):
    predictions = jax.vmap(
        jax.vmap(
            model, in_axes=(None, 0)  # vectorize over trunk(s)
        ), in_axes=(0, None)  # vectorize over branch(es)
    )(branch_in, trunk_in)
    rmse = jnp.sqrt(jnp.mean(jnp.square(predictions - target)))
    return rmse


@eqx.filter_jit
def training_step(model, branch_in, trunk_in, target, opt, opt_state):
    loss, grads = loss_fn(model, branch_in, trunk_in, target)
    updates, new_state = opt.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss, grads


def import_antiderivative_data(nqubits):
    rootdir = os.getcwd()
    os.chdir("../antiderivative_data")
    sys.path.append(os.getcwd())
    from load_antiderivative_data import load_antiderivative_data_deeponet
    branch_inputs_train, branch_inputs_test, trunk_inputs_train, trunk_inputs_test, outputs_train, outputs_test = load_antiderivative_data_deeponet(ngrid=nqubits)
    os.chdir(rootdir)
    return branch_inputs_train, branch_inputs_test, trunk_inputs_train, trunk_inputs_test, outputs_train, outputs_test


def import_heat_eqn_data(nqubits):
    rootdir = os.getcwd()
    os.chdir("../heat_eqn_data")
    sys.path.append(os.getcwd())
    from load_heat_eqn_data import load_heat_eqn_data_deeponet
    branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn = load_heat_eqn_data_deeponet(ngrid=nqubits)
    os.chdir(rootdir)
    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn


def import_burgers_data(nqubits, nquery, modstr=""):
    MM = (NGRID - 1) * 10 + 1
    rootdir = os.getcwd()
    os.chdir("../burgers_data")
    sys.path.append(os.getcwd())
    from load_burgers_data import load_burgers_data_deeponet
    branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = load_burgers_data_deeponet(NMESH=MM)
    os.chdir(rootdir)
    return branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test


def train_model(mode:str):
    print("==> NGRID has been set to", NGRID)
    match mode.lower():
        case "antiderivative":
            branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_antiderivative_data(NGRID)
        case "heat":
            branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_heat_eqn_data(NGRID)
        case "burgers":
            MM = (NGRID - 1) * 10 + 1
            MMM = f"_{MM}"
            branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_burgers_data(NGRID, nquery=NGRID, modstr=MMM)
        case _:
            sys.exit(1)

    EQUIV = 0
    if len(sys.argv) == 3:
        EQUIV = int(sys.argv[2])

    match EQUIV:
        case 0:
            # Equivalent Dimensions: Depth is NGRID
            print("Equivalent Dimension Neural Network")
            operator = DeepONet1D(
                in_size_branch=NGRID,
                branch_width=NGRID,
                branch_depth=NGRID,
                trunk_width=NGRID,
                trunk_depth=NGRID,
                interact_size=NGRID,
                activation=jax.nn.relu,
                key=jr.PRNGKey(0)
            )
        case 1:
            # Equivalent Number of Operations
            print("Equivalent Number of Operations")
            operator = DeepONet1D(
                in_size_branch=NGRID,
                branch_width=NGRID,
                branch_depth=2,
                trunk_width=NGRID,
                trunk_depth=2,
                interact_size=NGRID,
                activation=jax.nn.relu,
                key=jr.PRNGKey(0)
            )
        case 2:
            # Equivalent Number of Operations on State Vector
            print("Equivalent Number of Operations on State Vector")
            operator = DeepONet1D(
                in_size_branch=NGRID,
                branch_width=2**(NGRID-2),
                branch_depth=2,
                trunk_width=2**(NGRID-2),
                trunk_depth=2,
                interact_size=NGRID,
                activation=jax.nn.relu,
                key=jr.PRNGKey(0)
            )

    resultdir = f"{mode.lower()}_{NGRID}_{EQUIV}_results"
    paramfile = f"{resultdir}/params.json"
    if not os.path.isdir(resultdir):
        os.mkdir(resultdir)

    optimizer = optax.adam(1e-3)
    # set optimizer state, eqx.is_array is also why the bias term is dim (1,)
    opt_state = optimizer.init(eqx.filter(operator, eqx.is_array))

    def save_history(history, fname):
        with open(fname, "w") as f:
            f.write("mean, std\n")
            for epoch in history:
                tmp = []
                for lss in epoch:
                    tmp.append(jnp.linalg.norm(lss))
                f.write(f"{jnp.mean(jnp.array(tmp))}, {jnp.std(jnp.array(tmp))}\n")

    # Training Loop
    loss_history = []
    grad_history = {
        "branch": [],
        "trunk": [],
        "bias": []
    }

    for epoch in tqdm(range(1500)):
        loss_history.append([])
        for key in grad_history.keys():
            grad_history[key].append([])

        for branch_in, trunk_in, target in deeponet_dataloader(branch_inputs_train, trunk_inputs_query, outputs_train, batchsize=32, epoch=epoch):
            operator, opt_state, loss, grads = training_step(
                operator, 
                branch_in,
                trunk_in,
                target,
                optimizer,
                opt_state
            )
            # breakpoint()
            
            loss_history[epoch].append(loss)
            #grad_history["branch"][epoch].append(
            #    jnp.concat((
            #        jnp.concat([lay.weight.ravel() for lay in grads.branch_net.layers]).ravel(),
            #        jnp.concat([lay.bias.ravel() for lay in grads.branch_net.layers]).ravel(),
            #    ))
            #)
            #grad_history["trunk"][epoch].append(
            #    jnp.concat((
            #        jnp.concat([lay.weight.ravel() for lay in grads.trunk_net.layers]).ravel(),
            #        jnp.concat([lay.bias.ravel() for lay in grads.trunk_net.layers]).ravel(),
            #    ))
            #)
            #grad_history["bias"][epoch].append(
            #    grads.bias[0]
            #)

        last_epoch_avg_loss = sum(loss_history[-1]) / len(loss_history[-1])
        if epoch % 10 == 0:
            print(f"End of epoch {epoch}, avg loss: {last_epoch_avg_loss:7.5f}")

    epoch_avg_loss = [sum(arr)/len(arr) for arr in loss_history]
    print("Last Epoch Loss:", epoch_avg_loss[-1])

    save_history(loss_history, f"{resultdir}/loss_history.txt")
    for key in grad_history.keys():
        save_history(grad_history[key], f"{resultdir}/grad_{key}.txt")

    plt.figure(figsize=(10,5))
    plt.semilogy(epoch_avg_loss)
    plt.grid()
    plt.ylabel("Training Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss vs Epoch")
    plt.savefig(f"{resultdir}/training_history.png")
    plt.close()    

    predictions_test = jax.vmap(
        jax.vmap(
            operator, in_axes=(None, 0)  # vectorize over trunk(s)
        ), in_axes=(0, None)  # vectorize over branch(es)
    )(branch_inputs_test, trunk_inputs_query)

    errs = jax.vmap(rmse_error)(predictions_test, outputs_test)

    with open(f"{resultdir}/test_report.txt", "w") as f:
        f.write(f"Mean RMSE over test set: {jnp.mean(errs)}\n")
        f.write(f"Standard Deviation of RMSE over test set: {jnp.std(errs)}\n")
        f.write(f"Median RMSE over test set: {jnp.median(errs)}\n")

    def viz(point):
        plt.figure(figsize=(10,5))
        plt.plot(trunk_inputs_branch[:,0], branch_inputs_test[point, :], marker="o", label="INPUT")
        plt.plot(trunk_inputs_query[:,0], outputs_test[point, :], marker="+", label="TARGET")
        plt.plot(trunk_inputs_query[:,0], predictions_test[point, :], marker="+", label="PREDICTION")
        rmse = jnp.sqrt(jnp.mean((outputs_test[point, :] - predictions_test[point, :])**2))
        plt.title(f"Deep O Net, Test Datapoint {point}, RMSE {rmse:.4f}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{resultdir}/test_{point}.png")
        plt.close()

    viz(13)
    viz(32)
    viz(42)
    viz(99)
    viz(117)
    viz(197)
    viz(222)
    viz(333)

    breakpoint()


if __name__ == "__main__":
    train_model(mode=["antiderivative", "heat", "burgers"][2])


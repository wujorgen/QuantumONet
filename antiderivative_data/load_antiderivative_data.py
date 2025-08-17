import jax.numpy as jnp
import os


def load_antiderivative_data_deeponet(ngrid:int=-1):
    """Loads aligned antiderivative data
    Args:
        ngrid (int): number of grid points to downsample to. default: -1 (no downsampling)
    Returns:
        out (tuple[jax.Array]): branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn
    """
    this_location = os.path.dirname(os.path.abspath(__file__))
    dataset_train = jnp.load(os.path.join(this_location, "antiderivative_aligned_train.npz"), allow_pickle=True)
    dataset_test = jnp.load(os.path.join(this_location, "antiderivative_aligned_test.npz"), allow_pickle=True)

    branch_inputs_train_raw = dataset_train["X"][0]
    trunk_inputs_train = dataset_train["X"][1]
    outputs_train_raw = dataset_train["y"]

    branch_inputs_test_raw = dataset_test["X"][0]
    trunk_inputs_test = dataset_test["X"][1]
    outputs_test_raw = dataset_test["y"]

    print(f"{branch_inputs_train_raw.min()=}, {branch_inputs_train_raw.max()=}, {branch_inputs_test_raw.min()=}, {branch_inputs_test_raw.max()=}")
    print(f"{outputs_train_raw.min()=}, {outputs_train_raw.max()=}, {outputs_test_raw.min()=}, {outputs_test_raw.max()=}")

    SCALE_FACTOR = 4
    scale_around_zero = lambda x: x/SCALE_FACTOR

    branch_inputs_train = scale_around_zero(branch_inputs_train_raw)
    branch_inputs_test = scale_around_zero(branch_inputs_test_raw)
    outputs_train = scale_around_zero(outputs_train_raw)
    outputs_test = scale_around_zero(outputs_test_raw)

    LOWER_BOUND = -1.1
    UPPER_BOUND = 1.1
    assert branch_inputs_train.min() > LOWER_BOUND and branch_inputs_test.min() > LOWER_BOUND, (branch_inputs_train.min(), branch_inputs_test.min())
    assert branch_inputs_train.max() <= UPPER_BOUND and branch_inputs_test.max() <= UPPER_BOUND, (branch_inputs_train.max(), branch_inputs_test.max())
    assert outputs_train.min() > LOWER_BOUND and outputs_test.min() > LOWER_BOUND, (outputs_train.min(), outputs_test.min())
    assert outputs_train.max() <= UPPER_BOUND and outputs_test.max() <= UPPER_BOUND, (outputs_train.max(), outputs_test.max())

    print(f"{branch_inputs_train.shape=}, {trunk_inputs_train.shape=}, {outputs_train.shape=}")
    print(f"{branch_inputs_test.shape=}, {trunk_inputs_test.shape=}, {outputs_test.shape=}")

    if ngrid > 0:
        DOWNSAMPLE = 1
        while branch_inputs_train[:, ::DOWNSAMPLE].shape[1] > ngrid:
            DOWNSAMPLE += 1
        print(f"{DOWNSAMPLE=} for {branch_inputs_train[:, ::DOWNSAMPLE].shape[1]} grid points")
    else:
        DOWNSAMPLE = 1

    branch_inputs_train_dwn = branch_inputs_train[:, ::DOWNSAMPLE]
    branch_inputs_test_dwn = branch_inputs_test[:, ::DOWNSAMPLE]
    trunk_inputs_train_dwn = trunk_inputs_train[::DOWNSAMPLE]
    trunk_inputs_test_dwn = trunk_inputs_test[::DOWNSAMPLE]
    outputs_train_dwn = outputs_train[:, ::DOWNSAMPLE]
    outputs_test_dwn = outputs_test[:, ::DOWNSAMPLE]

    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn


def load_antiderivative_data(ngrid:int=-1):
    this_location = os.path.dirname(os.path.abspath(__file__))
    dataset_train = jnp.load(os.path.join(this_location, "antiderivative_aligned_train.npz"), allow_pickle=True)
    dataset_test = jnp.load(os.path.join(this_location, "antiderivative_aligned_test.npz"), allow_pickle=True)

    branch_inputs_train_raw = dataset_train["X"][0]
    trunk_inputs_train = dataset_train["X"][1]
    outputs_train_raw = dataset_train["y"]

    branch_inputs_test_raw = dataset_test["X"][0]
    trunk_inputs_test = dataset_test["X"][1]
    outputs_test_raw = dataset_test["y"]

    SCALE_FACTOR = 4
    scale_around_zero = lambda x: x/SCALE_FACTOR

    branch_inputs_train = scale_around_zero(branch_inputs_train_raw)
    branch_inputs_test = scale_around_zero(branch_inputs_test_raw)
    outputs_train = scale_around_zero(outputs_train_raw)
    outputs_test = scale_around_zero(outputs_test_raw)

    if ngrid > 0:
        DOWNSAMPLE = 1
        while branch_inputs_train[:, ::DOWNSAMPLE].shape[1] > ngrid:
            DOWNSAMPLE += 1
        print(f"{DOWNSAMPLE=} for {branch_inputs_train[:, ::DOWNSAMPLE].shape[1]} grid points")
    else:
        DOWNSAMPLE = 1
    
    branch_inputs_train_dwn = branch_inputs_train[:, ::DOWNSAMPLE]
    branch_inputs_test_dwn = branch_inputs_test[:, ::DOWNSAMPLE]
    trunk_inputs_train_dwn = trunk_inputs_train[::DOWNSAMPLE]
    trunk_inputs_test_dwn = trunk_inputs_test[::DOWNSAMPLE]
    outputs_train_dwn = outputs_train[:, ::DOWNSAMPLE]
    outputs_test_dwn = outputs_test[:, ::DOWNSAMPLE]

    return branch_inputs_train_dwn, branch_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn

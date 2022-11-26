import numpy as np

from pyma.oma import ssi
from pyma.preprocess import block_hankel
from pyma.postprocess import ss_to_modal, modal_assurance


def test_covariances(default_oma):

    y, ssm = default_oma

    opts = {
        "max_model_order": 10,  # N. Modes must be less than or equal hankel_dims/4
        "model_order": -1,
        "dt": 1e-2,
    }

    p = y.shape[0]

    alg = ssi.SSI(opts)
    hankel_dims = opts["max_model_order"] * 2
    Y = block_hankel(y, hankel_dims)
    Spp, Sfp, Sff = alg._compute_covariances(Y, p)

    assert np.allclose(
        Y[: opts["max_model_order"] * p, :].dot(Y[: opts["max_model_order"] * p, :].T),
        Spp,
    )
    assert np.allclose(
        Y[opts["max_model_order"] * p :, :].dot(Y[opts["max_model_order"] * p :, :].T),
        Sff,
    )
    assert np.allclose(
        Y[opts["max_model_order"] * p :, :].dot(Y[: opts["max_model_order"] * p, :].T),
        Sfp,
    )


def test_ssi(default_oma):

    y, ssm = default_oma
    true_props = ss_to_modal(ssm.A, ssm.C, 1e-2)

    opts = {
        "max_model_order": 10,  # N. Modes must be less than or equal hankel_dims/4
        "model_order": -1,
        "dt": 1e-2,
    }

    alg = ssi.SSI(opts)

    for f in opts.keys():
        assert alg.opts[f] == opts[f]

    props = alg(y)

    assert len(props) == opts["max_model_order"]

    # Do all my modal properties come out the right shape at least
    for i, p in enumerate(props):
        assert len(p[0]) == (i + 1) * y.shape[0]
        assert len(p[1]) == (i + 1) * y.shape[0]
        assert p[2].shape[1] == (i + 1) * y.shape[0]
        assert p[2].shape[0] == y.shape[0]

    # Frequency correctness, within 2% at correct model order
    assert np.all(np.abs((true_props[0] - props[1][0]) / true_props[0]) < 0.05)

    # Mode shape closeness at correct model order
    assert np.all(np.diag(modal_assurance(true_props[2], props[1][2])) > 0.95)

    # Damping is a mugs game so we won't even check

    opts.update({"model_order": 2})  # Two modes only
    alg = ssi.SSI(opts)

    props = alg(y)

    assert len(props) == 3
    assert len(props[0]) == 4
    assert len(props[1]) == 4
    assert props[2].shape[0] == 2
    assert props[2].shape[1] == 4

    # Frequency correctness, within 2% at correct model order
    assert np.all(np.abs((true_props[0] - props[0]) / true_props[0]) < 0.05)

    # Mode shape closeness at correct model order
    assert np.all(np.diag(modal_assurance(true_props[2], props[2])) > 0.95)

    opts.update({"model_order": [2, 4]})
    alg = ssi.SSI(opts)

    props = alg(y)

    # Frequency correctness, within 2% at correct model order
    assert np.all(np.abs((true_props[0] - props[0][0]) / true_props[0]) < 0.05)

    # Mode shape closeness at correct model order
    assert np.all(np.diag(modal_assurance(true_props[2], props[0][2])) > 0.95)

    # Do all my modal properties come out the right shape at least
    for i, p in enumerate(props):
        assert len(p[0]) == opts["model_order"][i] * y.shape[0]
        assert len(p[1]) == opts["model_order"][i] * y.shape[0]
        assert p[2].shape[1] == opts["model_order"][i] * y.shape[0]
        assert p[2].shape[0] == y.shape[0]

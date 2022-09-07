import pytest

import clairvoyance2.datasets.simulated.simple_pkpd
from clairvoyance2.datasets import simple_pkpd_dataset
from clairvoyance2.treatment_effects.synctwin import SyncTwinRegressor

# pylint: disable=protected-access


@pytest.mark.parametrize("time_index_treatment_event", [6, 3])
@pytest.mark.parametrize(
    "n_control_samples, n_treated_samples",
    [
        (8, 8),
        (8, 4),
    ],
)
def test_sanity_check_data_format(time_index_treatment_event, n_control_samples, n_treated_samples):
    clairvoyance2.datasets.simulated.simple_pkpd._SANITY_CHECK_ON = True
    data = simple_pkpd_dataset(
        n_timesteps=7,
        time_index_treatment_event=time_index_treatment_event,
        n_control_samples=n_control_samples,
        n_treated_samples=n_treated_samples,
        seed=100,
    )
    data_expected = clairvoyance2.datasets.simulated.simple_pkpd._sanity_check
    clairvoyance2.datasets.simulated.simple_pkpd._SANITY_CHECK_ON = False

    synctwin = SyncTwinRegressor()

    prepared_data, (n_treated_samples_, n_control_samples_) = synctwin._convert_data_to_synctwin_format(data)

    assert (prepared_data.x_full.numpy() == data_expected["x_full"]).all()
    assert (prepared_data.t_full.numpy() == data_expected["t_full"]).all()
    assert (prepared_data.mask_full.numpy() == data_expected["mask_full"]).all()
    assert (prepared_data.batch_ind_full.numpy() == data_expected["batch_ind_full"]).all()
    assert (prepared_data.y_full.numpy() == data_expected["y_full"]).all()
    assert (prepared_data.y_control.numpy() == data_expected["y_control"]).all()
    assert (prepared_data.y_mask_full.numpy() == data_expected["y_mask_full"]).all()
    assert n_control_samples == n_control_samples_
    assert n_treated_samples_ == n_treated_samples_

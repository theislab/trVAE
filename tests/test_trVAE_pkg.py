import trvae


def test_trVAE():
    network = trvae.archs.trVAEMulti(x_dimension=1000,
                                     n_conditions=3,
                                     z_dimension=15,
                                     dropout_rate=0.2)

    assert network.x_dim == 1000
    assert network.n_conditions == 3
    assert network.z_dim == 15
    assert network.dr_rate == 0.2

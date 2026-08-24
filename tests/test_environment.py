import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.constants import (
    AU,
    EARTH_DIPOLE_EQUATOR_FIELD,
    EARTH_GEOMAGNETIC_REFERENCE_RADIUS,
    EARTH_RADIUS,
    MOON_RADIUS,
)
from ssapy_toolkit.propagators_6dof import Spacecraft
from ssapy_toolkit.environment import (
    ALL_THIRD_BODY_NAMES,
    DEFAULT_THIRD_BODY_NAMES,
    FORCE_MODEL_PRESETS,
    PLANET_THIRD_BODY_NAMES,
    SpaceEnvironment,
    body_mu,
    body_radius,
    cylindrical_eclipse_fraction,
    earth_dipole_magnetic_field,
    exponential_atmosphere,
    igrf_magnetic_field,
    make_space_environment,
    solar_disk_visible_fraction,
    solar_occultation_fraction,
)
from ssapy_toolkit.satellites import Facet, MagneticDipole, SpacecraftBody
from ssapy_toolkit.accelerations_6dof import SpacecraftGravityGradientTorque


def test_environment_default_ephemerides_return_gcrf_vectors():
    environment = SpaceEnvironment()

    assert environment.sun_position(0.0).shape == (3,)
    assert environment.moon_position(0.0).shape == (3,)
    assert np.all(np.isfinite(environment.sun_position(0.0)))
    assert np.all(np.isfinite(environment.moon_position(0.0)))


def test_environment_epoch_offsets_relative_times_to_gps_seconds():
    epoch = Time("2025-01-01T00:00:00", scale="utc")
    environment = SpaceEnvironment(epoch=epoch)

    assert environment.absolute_time(10.0) == pytest.approx(epoch.gps + 10.0)
    assert environment.absolute_time(epoch) == pytest.approx(epoch.gps)
    assert SpaceEnvironment().absolute_time(10.0) == pytest.approx(10.0)


def test_environment_epoch_offsets_time_aware_callbacks():
    epoch = Time("2025-01-01T00:00:00", scale="utc")
    calls = []

    def vector_model(time, *_args):
        calls.append(time)
        return [1.0, 2.0, 3.0]

    def eclipse_model(time, *_args):
        calls.append(time)
        return 1.0

    environment = SpaceEnvironment(
        epoch=epoch,
        sun_position_model=vector_model,
        moon_position_model=vector_model,
        magnetic_field_model=vector_model,
        eclipse_model=eclipse_model,
    )

    environment.sun_position(1.0)
    environment.moon_position(2.0)
    environment.magnetic_field(3.0, [EARTH_RADIUS, 0.0, 0.0])
    environment.eclipse_fraction(4.0, [EARTH_RADIUS, 0.0, 0.0])

    np.testing.assert_allclose(
        calls,
        [epoch.gps + 1.0, epoch.gps + 2.0, epoch.gps + 3.0, epoch.gps + 4.0],
    )


def test_eclipse_models_return_full_partial_and_zero_sun_fraction():
    sun = np.array([AU, 0.0, 0.0])

    assert cylindrical_eclipse_fraction([2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(1.0)
    assert cylindrical_eclipse_fraction([-2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(0.0)
    assert solar_disk_visible_fraction([2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(1.0)
    assert solar_disk_visible_fraction([-2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(0.0)

    partial = solar_disk_visible_fraction([-2.0 * EARTH_RADIUS, EARTH_RADIUS, 0.0], sun)
    assert 0.0 < partial < 1.0

    assert solar_occultation_fraction(
        [0.0, 0.0, 0.0],
        sun,
        [10.0 * MOON_RADIUS, 0.0, 0.0],
        MOON_RADIUS,
    ) == pytest.approx(0.0)
    assert solar_occultation_fraction(
        [0.0, 0.0, 0.0],
        sun,
        [10.0 * MOON_RADIUS, 10.0 * MOON_RADIUS, 0.0],
        MOON_RADIUS,
    ) == pytest.approx(1.0)


def test_environment_conical_eclipse_includes_moon_shadow_for_srp():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_facets(
        Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], cr=1.0),
        append=False,
    )
    spacecraft = Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], body=body)
    shadowed_environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[10.0 * MOON_RADIUS, 0.0, 0.0],
    )
    earth_only_environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[10.0 * MOON_RADIUS, 0.0, 0.0],
        solar_occulting_bodies=("earth",),
    )

    assert shadowed_environment.eclipse_fraction(0.0, spacecraft.r) == pytest.approx(0.0)
    assert earth_only_environment.eclipse_fraction(0.0, spacecraft.r) == pytest.approx(1.0)

    shadowed_srp = shadowed_environment.force_models(solar_radiation=True, body=body)[0]
    illuminated_srp = earth_only_environment.force_models(solar_radiation=True, body=body)[0]
    np.testing.assert_allclose(shadowed_srp(spacecraft), 0.0, atol=1e-20)
    assert np.linalg.norm(illuminated_srp(spacecraft)) > 0.0


def test_environment_builds_force_models_with_state_aware_density():
    calls = []
    velocity_calls = []

    def density(altitude, time, r, v, q, omega, spacecraft):
        calls.append((altitude, time, spacecraft.mass))
        return 1.0e-12

    def atmosphere_velocity(time, r, v, q, omega, spacecraft):
        velocity_calls.append((time, spacecraft.mass))
        return v

    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_facets(
        Facet(area=2.0, normal_body=[1.0, 0.0, 0.0]),
        append=False,
    ).with_magnetic_dipoles(
        MagneticDipole(moment_body=[1.0, 0.0, 0.0]),
    )
    spacecraft = Spacecraft(
        r=[EARTH_RADIUS + 400_000.0, 0.0, 0.0],
        v=[0.0, 7_700.0, 0.0],
        body=body,
    )
    environment = make_space_environment(
        sun_position_model=[AU, 0.0, 0.0],
        atmosphere_density_model=density,
        atmosphere_velocity_model=atmosphere_velocity,
        magnetic_field_model=[0.0, 1.0e-5, 0.0],
        eclipse_model=None,
    )

    models = environment.force_models(drag=True, solar_radiation=True, magnetic=True)

    assert len(models) == 3
    for model in models:
        value = model(spacecraft)
        assert np.asarray(value).shape == (3,)
    assert calls
    assert calls[0][0] == pytest.approx(400_000.0)
    assert calls[0][1] == pytest.approx(spacecraft.t)
    assert calls[0][2] == pytest.approx(spacecraft.mass)
    assert velocity_calls == [(spacecraft.t, spacecraft.mass)]


def test_earth_dipole_magnetic_field_matches_equator_and_pole_limits():
    reference_radius = EARTH_GEOMAGNETIC_REFERENCE_RADIUS

    np.testing.assert_allclose(
        earth_dipole_magnetic_field([reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        earth_dipole_magnetic_field([0.0, 0.0, reference_radius]),
        [0.0, 0.0, 2.0 * EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        earth_dipole_magnetic_field([2.0 * reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD / 8.0],
    )

    environment = SpaceEnvironment()
    np.testing.assert_allclose(
        environment.magnetic_field(0.0, [reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        SpaceEnvironment(magnetic_field_model="zero").magnetic_field(
            0.0,
            [reference_radius, 0.0, 0.0],
        ),
        0.0,
    )
    with pytest.raises(ValueError, match="r_inertial"):
        environment.magnetic_field(0.0)


def test_environment_igrf_model_is_optional_and_returns_tesla_when_available():
    environment = SpaceEnvironment(magnetic_field_model="igrf")
    position = [EARTH_RADIUS + 400_000.0, 0.0, 0.0]

    try:
        field = environment.magnetic_field("2025-01-01T00:00:00", position)
    except ImportError as exc:
        assert "ppigrf" in str(exc)
    else:
        assert field.shape == (3,)
        assert np.linalg.norm(field) < 1.0e-3
        np.testing.assert_allclose(
            field,
            igrf_magnetic_field("2025-01-01T00:00:00", position),
        )

    with pytest.raises(ValueError, match="r_inertial"):
        environment.magnetic_field(0.0)


def test_environment_builds_third_body_models_from_named_bodies():
    spacecraft = Spacecraft(
        r=[EARTH_RADIUS + 700_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        body=SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)),
    )
    environment = SpaceEnvironment(moon_position_model=[384_400_000.0, 0.0, 0.0])

    models = environment.force_models(third_bodies="moon")

    assert len(models) == 1
    np.testing.assert_allclose(
        models[0](spacecraft),
        models[0].acceleration(
            t=spacecraft.t,
            r=spacecraft.r,
            v=spacecraft.v,
            q=spacecraft.q,
            omega=spacecraft.omega,
            spacecraft=spacecraft,
        ),
    )
    assert np.linalg.norm(models[0](spacecraft)) > 0.0
    assert body_mu("moon") > 0.0
    assert body_mu("Sun") > body_mu("moon")


def test_environment_third_body_selector_presets_are_explicit():
    assert DEFAULT_THIRD_BODY_NAMES == ("moon", "sun")
    assert "earth" not in PLANET_THIRD_BODY_NAMES
    assert ALL_THIRD_BODY_NAMES == DEFAULT_THIRD_BODY_NAMES + PLANET_THIRD_BODY_NAMES


def test_environment_builds_all_planet_third_body_models():
    environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[384_400_000.0, 0.0, 0.0],
    )

    default_models = environment.force_models(third_bodies=True)
    planet_models = environment.force_models(third_bodies="planets")
    all_models = environment.force_models(third_bodies="all")

    assert len(default_models) == len(DEFAULT_THIRD_BODY_NAMES)
    assert len(planet_models) == len(PLANET_THIRD_BODY_NAMES)
    assert len(all_models) == len(ALL_THIRD_BODY_NAMES)


def test_environment_force_model_presets_are_user_friendly():
    environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[384_400_000.0, 0.0, 0.0],
        atmosphere_density_model=1.0e-12,
        magnetic_field_model=[0.0, 0.0, 1.0e-5],
        eclipse_model=None,
    )

    assert {"none", "earth_orbit", "leo", "cislunar", "all"} <= set(FORCE_MODEL_PRESETS)
    assert len(environment.force_models(preset="none")) == 0
    assert len(environment.force_models(preset="earth_orbit")) == 4
    assert len(environment.force_models(preset="leo")) == 6
    assert len(environment.force_models(preset="cislunar")) == 13
    assert len(environment.force_models(preset="all", drag=False)) == 14


def test_environment_force_model_preset_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown force model preset"):
        SpaceEnvironment().force_models(preset="kitchen_sink")


def test_environment_builds_gravity_gradient_torque_models():
    body = SpacecraftBody(
        name="bus",
        mass=10.0,
        inertia=np.diag([10.0, 20.0, 30.0]),
    )
    spacecraft = Spacecraft(
        r=[7_000_000.0, 7_000_000.0, 0.0],
        v=[0.0, 0.0, 0.0],
        body=body,
    )
    environment = SpaceEnvironment(
        moon_position_model=[384_400_000.0, 0.0, 0.0],
        sun_position_model=[AU, 0.0, 0.0],
    )

    earth_models = environment.force_models(gravity_gradient=True)
    assert len(earth_models) == 1
    assert isinstance(earth_models[0], SpacecraftGravityGradientTorque)
    assert np.linalg.norm(earth_models[0](spacecraft)) > 0.0

    all_models = environment.force_models(gravity_gradient="all")
    assert len(all_models) == 3
    assert all(isinstance(model, SpacecraftGravityGradientTorque) for model in all_models)


def test_exponential_atmosphere_validates_and_decays():
    density = exponential_atmosphere(
        reference_density=1.0e-12,
        reference_altitude=400_000.0,
        scale_height=50_000.0,
    )

    assert density(400_000.0) == pytest.approx(1.0e-12)
    assert density(450_000.0) < density(400_000.0)
    with pytest.raises(ValueError, match="positive"):
        exponential_atmosphere(reference_density=0.0, reference_altitude=0.0, scale_height=1.0)


def test_environment_top_level_aliases():
    import ssapy_toolkit as ssatk

    assert ssatk.SpaceEnvironment is SpaceEnvironment
    assert ssatk.DEFAULT_THIRD_BODY_NAMES == DEFAULT_THIRD_BODY_NAMES
    assert ssatk.PLANET_THIRD_BODY_NAMES == PLANET_THIRD_BODY_NAMES
    assert ssatk.ALL_THIRD_BODY_NAMES == ALL_THIRD_BODY_NAMES
    assert ssatk.FORCE_MODEL_PRESETS == FORCE_MODEL_PRESETS
    assert ssatk.make_space_environment is make_space_environment
    assert ssatk.exponential_atmosphere is exponential_atmosphere
    assert ssatk.earth_dipole_magnetic_field is earth_dipole_magnetic_field
    assert ssatk.igrf_magnetic_field is igrf_magnetic_field
    assert ssatk.solar_disk_visible_fraction is solar_disk_visible_fraction
    assert ssatk.solar_occultation_fraction is solar_occultation_fraction
    assert ssatk.body_mu is body_mu
    assert ssatk.body_radius is body_radius
    assert body_radius("moon") == pytest.approx(MOON_RADIUS)

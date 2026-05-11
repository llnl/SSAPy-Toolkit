import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos.demo_all_orbit_quantities import main as demo_all_orbit_quantities
from demos.demo_build_dashboard import main as demo_build_dashboard
from demos.demo_compare_models import main as demo_compare_models
from demos.demo_converting_impulse_and_burns import main as demo_converting_impulse_and_burns
from demos.demo_ellipse_ae_for_arrival_rv import main as demo_ellipse_ae_for_arrival_rv
from demos.demo_ellipses import main as demo_ellipses
from demos.demo_gcrs_to_itrs_astropy import main as demo_gcrs_to_itrs_astropy
from demos.demo_groundtrack_accuracy import main as demo_groundtrack_accuracy
from demos.demo_groundtrack_plot import main as demo_groundtrack_plot
from demos.demo_kepler_vs_harmonics import main as demo_kepler_vs_harmonics
from demos.demo_orbital_stats_dashboard import main as demo_orbital_stats_dashboard
from demos.demo_parsing_3le import main as demo_parsing_3le
from demos.demo_sampling import main as demo_sampling
from demos.demo_sphere_generation import main as demo_sphere_generation
from demos.demo_ssapy_ground_lambertian_reflectance import main as demo_ssapy_ground_lambertian_reflectance
from demos.demo_ssapy_orbit_and_plot import main as demo_ssapy_orbit_and_plot
from demos.demo_transfer_rendezvous import main as demo_transfer_rendezvous


def test_demo_all_orbit_quantities():
    out = demo_all_orbit_quantities(verbose=False)
    assert "rv_case" in out
    assert "rpra_case" in out
    assert "ae_ma_case" in out


def test_demo_build_dashboard():
    out = demo_build_dashboard(make_figures=False, fast=True)
    assert "fig" in out


def test_demo_compare_models():
    out = demo_compare_models(make_figures=False, fast=True)
    assert "out" in out
    assert "figure_time_domain" in out
    assert "figure_rung_summary" in out


def test_demo_converting_impulse_and_burns():
    out = demo_converting_impulse_and_burns(make_figures=False, fast=True)
    assert "burn_to_deltav" in out
    assert "deltav_to_burn" in out


def test_demo_ellipse_ae_for_arrival_rv():
    out = demo_ellipse_ae_for_arrival_rv(make_figures=False, verbose=False)
    assert "result" in out


def test_demo_ellipses():
    out = demo_ellipses(make_figures=False, fast=True)
    assert "trajectories" in out
    assert len(out["trajectories"]) > 0


def test_demo_gcrs_to_itrs_astropy():
    out = demo_gcrs_to_itrs_astropy(verbose=False)
    assert "itrs_coords" in out
    assert "itrs_position" in out


def test_demo_groundtrack_accuracy():
    out = demo_groundtrack_accuracy(make_figures=False, make_video=False, fast=True)
    assert "dr_norm_km" in out
    assert len(out["dr_norm_km"]) > 0


def test_demo_groundtrack_plot():
    out = demo_groundtrack_plot(make_figures=False, fast=True)
    assert "tracks" in out
    assert len(out["tracks"]) == 3


def test_demo_kepler_vs_harmonics():
    out = demo_kepler_vs_harmonics(fast=True)
    assert "errs_km" in out
    assert len(out["errs_km"]) > 0


def test_demo_orbital_stats_dashboard():
    out = demo_orbital_stats_dashboard(make_figures=False, fast=True)
    assert "population" in out or "figure" in out


def test_demo_parsing_3le():
    out = demo_parsing_3le(verbose=False, fast=True)
    assert "data" in out
    if out.get("skipped"):
        assert out["reason"] == "missing_data_file"
    else:
        assert out["data"] is not None


def test_demo_sampling():
    out = demo_sampling(make_figures=False, fast=True, verbose=False)
    assert "Uniform ball" in out
    assert "Gaussian" in out


def test_demo_sphere_generation():
    out = demo_sphere_generation(make_figures=False, fast=True)
    assert "uniform" in out
    assert "random" in out


def test_demo_ssapy_ground_lambertian_reflectance():
    out = demo_ssapy_ground_lambertian_reflectance(make_figures=False, fast=True)
    assert "mv" in out
    assert len(out["mv"]) > 0


def test_demo_ssapy_orbit_and_plot():
    out = demo_ssapy_orbit_and_plot(make_figures=False, fast=True)
    assert "r" in out
    assert "mv" in out


def test_demo_transfer_rendezvous():
    out = demo_transfer_rendezvous(make_figures=False)
    assert "|delta_v1|" in out
    assert "|delta_v2|" in out
"""Tests for per-band template coverage maps.

Coverage maps are produced upstream and published as one HEALPix FITS file per
band. The real files live on a shared filesystem that is not mounted here, so
these tests write equivalent files with `healpy` and exercise the same reader.
"""

import json

import healpy as hp
import numpy as np
import pytest

from chatterbox.astro.skymap import localization_from_probability
from chatterbox.astro.templates import (
    BANDS,
    DEFAULT_MAP_PATTERN,
    TemplateCoverage,
    load_source_maps,
    load_template_maps,
    map_filename,
    read_coverage_map,
)

NSIDE = 64


def disc_map(ra=60.0, dec=-35.0, radius=5.0, nside=NSIDE, value=1.0):
    """A coverage map that is `value` inside a disc and 0 elsewhere."""
    m = np.zeros(hp.nside2npix(nside))
    m[hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius), inclusive=True)] = value
    return m


def write_maps(directory, bands=("g", "i"), nside=NSIDE, nest=False, **disc_kwargs):
    """Write per-band FITS coverage maps using the published naming scheme."""
    directory.mkdir(parents=True, exist_ok=True)
    for band in bands:
        m = disc_map(nside=nside, **disc_kwargs)
        if nest:
            m = hp.reorder(m, r2n=True)
        hp.write_map(
            directory / map_filename(band, nside),
            m,
            nest=nest,
            overwrite=True,
            dtype=np.float32,
        )
    return directory


def point_localization(ra=60.0, dec=-35.0, radius=1.0, nside=NSIDE):
    prob = np.zeros(hp.nside2npix(nside))
    prob[hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius), inclusive=True)] = 1.0
    return localization_from_probability(prob, provenance="test")


# ------------------------------------------------------------------ filenames


def test_map_filename_matches_the_published_convention():
    assert map_filename("y", 64) == "template_coverage_healpix_y_nside64.fits"
    assert map_filename("u", 128) == "template_coverage_healpix_u_nside128.fits"
    assert "{band}" in DEFAULT_MAP_PATTERN and "{nside}" in DEFAULT_MAP_PATTERN


def test_map_filename_honours_a_custom_pattern():
    assert map_filename("r", 64, "cov_{nside}_{band}.fits") == "cov_64_r.fits"


# --------------------------------------------------------------------- source


def test_load_source_maps_reads_each_band(tmp_path):
    write_maps(tmp_path / "out", bands=("g", "i"))
    coverage = load_source_maps(tmp_path / "out", nside=NSIDE)

    assert coverage.bands == ["g", "i"]
    assert coverage.nside == NSIDE
    assert coverage.maps["g"].size == hp.nside2npix(NSIDE)
    assert coverage.area_deg2("g") > 0
    assert str(tmp_path / "out") in coverage.source
    assert coverage.built_at


def test_all_six_bands_are_read_when_published(tmp_path):
    write_maps(tmp_path / "out", bands=BANDS)
    coverage = load_source_maps(tmp_path / "out", nside=NSIDE)
    assert coverage.bands == list(BANDS)
    assert coverage.missing_bands == []


def test_missing_band_is_skipped_not_fatal(tmp_path, caplog):
    """Coverage is built up band by band upstream, so gaps are normal."""
    write_maps(tmp_path / "out", bands=("g",))
    with caplog.at_level("WARNING"):
        coverage = load_source_maps(tmp_path / "out", nside=NSIDE)
    assert coverage.bands == ["g"]
    assert set(coverage.missing_bands) == set(BANDS) - {"g"}
    assert "u-band" in caplog.text


def test_missing_directory_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="templates.maps_dir"):
        load_source_maps(tmp_path / "absent", nside=NSIDE)


def test_directory_with_no_matching_maps_raises(tmp_path):
    empty = tmp_path / "out"
    empty.mkdir()
    (empty / "unrelated.fits").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="No template coverage maps"):
        load_source_maps(empty, nside=NSIDE)


def test_wrong_nside_file_is_rejected(tmp_path, caplog):
    """A file whose pixel count disagrees with its name must not be used."""
    out = tmp_path / "out"
    out.mkdir()
    # Name says nside 64, contents are nside 32.
    hp.write_map(out / map_filename("g", NSIDE), disc_map(nside=32), overwrite=True)
    with caplog.at_level("ERROR"):
        with pytest.raises(FileNotFoundError):
            load_source_maps(out, nside=NSIDE)
    assert "expected" in caplog.text


def test_nested_files_are_converted_to_ring(tmp_path):
    """healpy honours the ORDERING keyword; both must agree."""
    write_maps(tmp_path / "ring", bands=("g",), nest=False)
    write_maps(tmp_path / "nest", bands=("g",), nest=True)

    ring = load_source_maps(tmp_path / "ring", nside=NSIDE)
    nested = load_source_maps(tmp_path / "nest", nside=NSIDE)
    assert nested.area_deg2("g") == pytest.approx(ring.area_deg2("g"), rel=1e-6)
    assert np.array_equal(nested.mask("g"), ring.mask("g"))


def test_read_coverage_map_returns_ring(tmp_path):
    out = write_maps(tmp_path / "out", bands=("g",))
    m = read_coverage_map(out / map_filename("g", NSIDE))
    assert m.size == hp.nside2npix(NSIDE)
    assert m.max() > 0


def test_custom_pattern_is_used(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    hp.write_map(out / "cov_64_g.fits", disc_map(), overwrite=True)
    coverage = load_source_maps(out, nside=NSIDE, pattern="cov_{nside}_{band}.fits")
    assert coverage.bands == ["g"]


# --------------------------------------------------------------------- binary


def test_maps_are_read_as_binary_coverage(tmp_path):
    """1 means a template exists, 0 means none."""
    write_maps(tmp_path / "out", bands=("z",), value=1.0)
    coverage = load_source_maps(tmp_path / "out", nside=NSIDE)
    assert set(np.unique(coverage.maps["z"])) == {0.0, 1.0}
    assert coverage.area_deg2("z") > 0
    assert coverage.mask("z").sum() == int((coverage.maps["z"] == 1).sum())


def test_non_binary_file_is_flagged_but_still_usable(tmp_path, caplog):
    """A format change upstream must be noticed, not silently reinterpreted."""
    write_maps(tmp_path / "out", bands=("g",), value=7.0)
    with caplog.at_level("WARNING"):
        coverage = load_source_maps(tmp_path / "out", nside=NSIDE)
    assert "not binary" in caplog.text
    # Non-zero still counts as covered, so the post is not silently empty.
    assert coverage.area_deg2("g") > 0


def test_nan_pixels_read_as_uncovered(tmp_path):
    """NaN must not raise or count as coverage."""
    m = disc_map()
    m[m == 0] = np.nan
    coverage = TemplateCoverage(maps={"r": m}, nside=NSIDE)
    assert not np.isnan(coverage.area_deg2("r"))
    assert coverage.mask("r").sum() == int(np.count_nonzero(np.nan_to_num(m)))


# ------------------------------------------------------------------- coverage


def test_coverage_in_region():
    coverage = TemplateCoverage(maps={"g": disc_map()}, nside=NSIDE)
    fractions = coverage.coverage_in_region(point_localization().prob_map, nside=NSIDE)
    assert fractions["g"] == pytest.approx(1.0, abs=0.02)
    assert fractions["r"] == 0.0
    assert set(fractions) == set(BANDS)


def test_coverage_in_region_resamples_across_resolutions():
    """The localization is usually at a different nside from the maps."""
    coverage = TemplateCoverage(maps={"g": disc_map()}, nside=NSIDE)
    fine = point_localization(nside=256)
    assert coverage.coverage_in_region(fine.prob_map, nside=256)["g"] == pytest.approx(1.0, abs=0.05)


def test_coverage_of_a_region_with_no_templates():
    coverage = TemplateCoverage(maps={"g": disc_map(ra=200.0, dec=10.0)}, nside=NSIDE)
    fractions = coverage.coverage_in_region(point_localization().prob_map, nside=NSIDE)
    assert fractions["g"] == pytest.approx(0.0, abs=1e-9)


def test_all_bands_reported_even_when_not_published():
    """A band with no map must report 0, not vanish from the post."""
    coverage = TemplateCoverage(maps={"g": disc_map()}, nside=NSIDE)
    fractions = coverage.coverage_in_region(point_localization().prob_map, nside=NSIDE)
    assert set(fractions) == set(BANDS)
    assert fractions["u"] == 0.0


# ---------------------------------------------------------------------- cache


def test_round_trip_through_the_cache(tmp_path):
    write_maps(tmp_path / "out", bands=("g", "i"))
    coverage = load_source_maps(tmp_path / "out", nside=NSIDE)
    coverage.save(tmp_path / "cache")
    loaded = load_template_maps(tmp_path / "cache")

    assert loaded is not None
    assert loaded.nside == coverage.nside
    assert loaded.bands == coverage.bands
    assert loaded.built_at == coverage.built_at
    assert loaded.source == coverage.source
    assert loaded.band_files == coverage.band_files
    for band in coverage.bands:
        assert np.allclose(loaded.maps[band], coverage.maps[band])


def test_missing_cache_returns_none_rather_than_raising(tmp_path):
    """A missing cache must degrade to a post without the template panel."""
    assert load_template_maps(tmp_path / "absent") is None


def test_corrupt_cache_returns_none(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text("{not json")
    assert load_template_maps(cache) is None


# ------------------------------------------------------------------------ CLI


def _config(tmp_path, **templates):
    path = tmp_path / "config.yaml"
    settings = {"cache_dir": str(tmp_path / "cache"), "nside": NSIDE}
    settings.update(templates)
    path.write_text(json.dumps({"templates": settings}))
    return path


def test_refresh_from_the_published_directory(tmp_path, capsys):
    """The full refresh path: published maps in, local cache out."""
    from chatterbox.cli import main

    write_maps(tmp_path / "out", bands=("g", "i"))
    config_path = _config(tmp_path, maps_dir=str(tmp_path / "out"))

    assert main(["-c", str(config_path), "refresh-templates"]) == 0

    coverage = load_template_maps(tmp_path / "cache")
    assert coverage is not None
    assert coverage.bands == ["g", "i"]
    assert coverage.nside == NSIDE
    out = capsys.readouterr().out
    assert "deg^2" in out
    assert "no map published for" in out


def test_refresh_reports_a_missing_directory(tmp_path, capsys):
    from chatterbox.cli import main

    config_path = _config(tmp_path, maps_dir=str(tmp_path / "absent"))
    assert main(["-c", str(config_path), "refresh-templates"]) == 1
    assert "templates.maps_dir" in capsys.readouterr().err


def test_refresh_path_override(tmp_path):
    """--path overrides the configured directory."""
    from chatterbox.cli import main

    write_maps(tmp_path / "configured", bands=("r",))
    write_maps(tmp_path / "override", bands=("z",))
    config_path = _config(tmp_path, maps_dir=str(tmp_path / "configured"))

    assert main(["-c", str(config_path), "refresh-templates", "--path", str(tmp_path / "override")]) == 0
    coverage = load_template_maps(tmp_path / "cache")
    assert coverage.bands == ["z"]


def test_refresh_honours_the_configured_bands(tmp_path):
    from chatterbox.cli import main

    write_maps(tmp_path / "out", bands=BANDS)
    config_path = _config(tmp_path, maps_dir=str(tmp_path / "out"), bands=["g", "r"])

    assert main(["-c", str(config_path), "refresh-templates"]) == 0
    assert load_template_maps(tmp_path / "cache").bands == ["g", "r"]

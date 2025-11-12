# Offline wheel cache

Place optional Python wheel files under `vendor/wheels/` when you need to run
the test Docker image without internet access. If wheels are present the build
uses `pip install --no-index --find-links /opt/wheels`; otherwise it installs
dependencies from PyPI (the default flow).

Recommended layout:

```
vendor/
  wheels/
    numpy-*.whl
    pandas-*.whl
    pyarrow-*.whl
    duckdb-*.whl
    pytest-*.whl
    ...
```

Keep wheel versions consistent with `docker/tests/requirements-tests.lock`.
When new dependencies are added, regenerate the lock file and refresh the
corresponding wheels before running offline builds. To force offline mode set
`ALLOW_ONLINE_INSTALL=0` (or add `--no-online-install` to `scripts/run_tests.sh`);
to skip dependency installation altogether set `ALLOW_EMPTY_WHEELS=1`.

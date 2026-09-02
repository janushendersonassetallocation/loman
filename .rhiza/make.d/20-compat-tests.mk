## .rhiza/make.d/20-compat-tests.mk - Cross-version compatibility checks
# This file is repo-owned: it is not in .rhiza/template.lock, so `make sync`
# leaves it alone.
#
# `make test` runs the suite against whatever the lock file resolves to, which
# is the newest supported pandas. Loman supports pandas 2 as well, and the two
# differ in the default temporal resolution --- nanoseconds versus microseconds.
# That difference is exactly what the serialization format records rather than
# assumes, so it needs a check that actually runs against the other end of the
# range; otherwise the support is accidental and breaks silently.
#
# The pin below is an exact series glob, not a lower bound. `--with "pandas>=2.0"`
# reads like "test the old one" and is not: uv resolves the newest version
# satisfying it, so both legs of the matrix installed the same pandas 2.3.3 and
# the job reported green while testing one version twice.
#
# Kept out of `make test` deliberately. It builds a separate environment, so
# folding it into the default target would roughly double the time of the loop
# people run most often. CI runs it as its own job (.github/workflows/pandas_compat.yml).

.PHONY: test-compat

# Oldest supported pandas series. Matches the floor declared in pyproject.toml,
# which is 2.3.3; this names the series so the pin below stays a two-part glob.
PANDAS_MIN ?= 2.3

# Only the modules whose behaviour depends on the pandas version. The rest of
# the suite is version-independent and already covered by `make test`. The
# property tests are included because the value model --- datetime resolution
# above all --- is precisely what differs between the two pandas versions.
#
# Deferred (`=`, not `:=`) because TESTS_FOLDER is defined in test.mk, which is
# included after this file: an immediate assignment would capture it empty and
# hand pytest a list of paths rooted at /.
COMPAT_TESTS = \
  $(TESTS_FOLDER)/test_serialization.py \
  $(TESTS_FOLDER)/test_serialization_behaviour.py \
  $(TESTS_FOLDER)/test_containers.py \
  $(TESTS_FOLDER)/test_compression.py \
  $(TESTS_FOLDER)/test_byos.py \
  $(TESTS_FOLDER)/test_pandas_compat.py \
  $(TESTS_FOLDER)/test_serialization_properties.py

##@ Loman Compatibility Testing

# Depends on install-uv rather than install: `uv run --isolated` resolves its
# own environment from pyproject.toml, so the project venv is not needed and
# building it first would only cost time.
test-compat: install-uv ## run the version-sensitive tests against the oldest supported pandas
	@printf "${BLUE}[Loman] Testing against pandas ~=$(PANDAS_MIN)...${RESET}\n"
	@$(UV_BIN) run --isolated \
	  --with "pandas==$(PANDAS_MIN).*" \
	  --with pytest \
	  --with hypothesis \
	  pytest $(COMPAT_TESTS) -q

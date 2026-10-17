# Non-editable wheel avoids CloudDocs .pth visibility problems. Always rebuild
# the local distribution before executing tests or commands after source edits.
UV ?= uv
export PYTHONDONTWRITEBYTECODE := 1
export UV_LINK_MODE := copy
export UV_PYTHON_DOWNLOADS := never
RUN_ROOT ?= $(abspath ../cbam-power-private/runs)
RUN_ID ?= run_$(shell date -u +%Y%m%dT%H%M%S)
UV_RUN = $(UV) run --locked --no-editable --extra weather --extra learning --group test

.PHONY: install demo test reproduce report notebooks release-check site site-check
install:
	$(UV) sync --locked --no-editable --extra weather --extra learning --group test --reinstall-package cbam-power

demo: install
	$(UV_RUN) cbam-power demo --output-root "$(RUN_ROOT)" --run-id "$(RUN_ID)"

test: install
	$(UV_RUN) python -B -m pytest -m "public and not authorized" tests -q
	@echo "Public/artificial tests only. Authorized frozen tests require explicit private inputs."

reproduce: install
	@test -n "$$CBAM_INPUT_ROOT" || (echo "BLOCKED_CBAM_INPUT_ROOT_REQUIRED"; exit 2)
	$(UV_RUN) cbam-power reproduce --mode authorized --output-root "$(RUN_ROOT)" --run-id "$(RUN_ID)"

report: install
	$(UV_RUN) cbam-power report --mode $(MODE) --audience $(AUDIENCE) --output-root "$(RUN_ROOT)" --run-id "$(RUN_ID)"
MODE ?= authorized
AUDIENCE ?= local

# Execute reviewed artificial notebooks into an external, new output directory.
notebooks: install
	$(UV_RUN) python -B -m cbam_power.portfolio --source-root "$(CURDIR)" --output-root "$(RUN_ROOT)/$(RUN_ID)"

# Checks source/artifact allowlists; this is not a publication rights grant.
release-check: install
	$(UV_RUN) python -B -m cbam_power.portfolio --source-root "$(CURDIR)" --output-root "$(RUN_ROOT)/$(RUN_ID)" --check-only $(if $(PACKAGE_ROOT),--packages "$(PACKAGE_ROOT)")

# One maintained Markdown narrative; offline, deterministic static HTML.
site: install
	$(UV_RUN) python -B -m cbam_power.site --source-root "$(CURDIR)"

site-check: install
	$(UV_RUN) python -B -m cbam_power.site --source-root "$(CURDIR)" --check

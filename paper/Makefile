.PHONY: test check check-protocols table-real fig-spf-emotions atlas contracted

PYTHON := python3
RESULTS := results
GENERATED := generated

test:
	PYTHONPATH=../src $(PYTHON) -m pytest ../tests --cov=msdpg -q

check-protocols:
	@for f in protocols/*/PROTOCOL.sha256; do \
		d=$$(dirname $$f); expected=$$(awk '{print $$1}' $$f); \
		actual=$$(sha256sum $$d/PROTOCOL.md | awk '{print $$1}'); \
		test "$$expected" = "$$actual" || { echo "checksum failed: $$d"; exit 1; }; \
		echo "$$d/PROTOCOL.md: OK"; \
	done

check: test check-protocols

table-real:
	$(PYTHON) scripts/make_table_real.py --input $(RESULTS)/t4_real.csv --output $(GENERATED)/table_real.tex

fig-spf-emotions:
	$(PYTHON) scripts/make_atlas.py --input $(RESULTS)/atlas/atlas.json --dataset emotions --output $(GENERATED)/spf_emotions.png

atlas:
	$(PYTHON) scripts/make_atlas.py --input $(RESULTS)/atlas/atlas.json --output $(GENERATED)/atlas.png

contracted:
	@echo "Frozen contracted-view summary: $(RESULTS)/contracted_view/summary.json"

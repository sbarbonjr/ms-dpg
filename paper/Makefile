.PHONY: test check table-real atlas contracted

test:
	PYTHONPATH=../src python3 -m pytest ../tests -q

# Stored artifacts are checked before computational reruns are added.
check: test
	@for f in protocols/*/PROTOCOL.sha256; do \
		d=$$(dirname $$f); expected=$$(awk '{print $$1}' $$f); \
		actual=$$(sha256sum $$d/PROTOCOL.md | awk '{print $$1}'); \
		test "$$expected" = "$$actual" || { echo "checksum failed: $$d"; exit 1; }; \
		echo "$$d/PROTOCOL.md: OK"; \
	done

table-real:
	@echo "Stored result: results/t4_real.csv"

atlas:
	@echo "Stored result: results/atlas/atlas.json"

contracted:
	@echo "Stored result: results/contracted_view/summary.json"

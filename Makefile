# Node-based workflow helpers for the dashboard

NPM ?= npm
NODE ?= node
PORT ?= 4100

.PHONY: help install start dev clean depclean

help:
	@echo "Targets:"
	@echo "  install  - Install npm dependencies"
	@echo "  start    - Run the Express server (same as npm start)"
	@echo "  dev      - Alias for start"
	@echo "  clean    - Remove node_modules"
	@echo "  depclean - Alias for clean"

install:
	$(NPM) install

start:
	PORT=$(PORT) $(NPM) start

dev: start

clean depclean:
	rm -rf node_modules

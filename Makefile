# NexaRAG — common tasks.
#
# `make` on its own prints this list. The one you want on a fresh clone is `make setup`.

APP_DIR := ../Nexarag-app
COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help setup setup-native setup-docker doctor up down restart logs ps shell db-shell \
        migrate revision seed dev test test-api test-app verify build clean reset

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- getting started --------------------------------------------------------------------
#
# On Windows there is no make: run `setup.cmd`, or `powershell -ExecutionPolicy Bypass
# -File scripts\bootstrap.ps1`. It installs what Windows is missing and then runs the same
# bootstrap.sh under Git Bash, so there is one implementation of the setup logic.

setup: ## Clone the SPA, write .env, and start everything (Docker, falling back to native)
	@./scripts/bootstrap.sh

setup-docker: ## Setup, refusing to fall back to a native install
	@./scripts/bootstrap.sh --docker

setup-native: ## Setup without Docker — venv + npm + MySQL, installing what is missing
	@./scripts/bootstrap.sh --native

doctor: ## Report what setup would do and what is missing, changing nothing
	@./scripts/bootstrap.sh --check

# --- Docker -----------------------------------------------------------------------------

up: ## Start the stack in the background
	$(COMPOSE) up -d

down: ## Stop the stack, keeping the database and uploaded documents
	$(COMPOSE) down

restart: ## Rebuild and restart after a code change
	$(COMPOSE) up -d --build

logs: ## Follow logs from every service
	$(COMPOSE) logs -f

ps: ## Show what is running
	$(COMPOSE) ps

shell: ## A shell inside the API container
	$(COMPOSE) exec api bash

db-shell: ## A MySQL prompt on the app database
	@$(COMPOSE) exec db sh -c 'exec mysql -u"$$MYSQL_USER" -p"$$MYSQL_PASSWORD" "$$MYSQL_DATABASE"'

# --- database ---------------------------------------------------------------------------

migrate: ## Apply migrations (the container does this on start; this is for a running stack)
	$(COMPOSE) exec api alembic upgrade head

revision: ## Autogenerate a migration — READ IT before applying (see CLAUDE.md §17)
	@test -n "$(m)" || { echo 'usage: make revision m="add_x_table"'; exit 1; }
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(m)"

seed: ## Seed the superadmin and default roles
	$(COMPOSE) exec api python -m seeders.user_seeder

# --- local embeddings (optional) ---------------------------------------------------------

embeddings: ## Start Ollama and pull EmbeddingGemma — free local embeddings, nothing leaves the machine
	$(COMPOSE) --profile embeddings up -d ollama
	@echo "pulling embeddinggemma (622 MB, once)…"
	$(COMPOSE) --profile embeddings exec ollama ollama pull embeddinggemma
	@echo
	@echo "Done. In the app: Knowledge Bases → upload → Embeddings → Ollama Local → embeddinggemma."

embeddings-down: ## Stop Ollama, keeping the downloaded model
	$(COMPOSE) --profile embeddings stop ollama

# --- development (native) ---------------------------------------------------------------

dev: ## Run both dev servers with hot reload, on the ports in .env (Ctrl-C stops both)
	@./scripts/dev.sh

# --- tests ------------------------------------------------------------------------------

test: test-api test-app ## Run both test suites

test-api: ## Backend suite (in-memory SQLite; no infrastructure needed)
	@. .venv/bin/activate 2>/dev/null && python -m pytest || python -m pytest

test-app: ## Frontend suite
	@cd $(APP_DIR) && npm test

verify: ## The gate before calling frontend work done: typecheck + coverage
	@cd $(APP_DIR) && npm run verify

# --- housekeeping -------------------------------------------------------------------------

build: ## Rebuild the images without starting anything
	$(COMPOSE) build

clean: ## Stop the stack and remove build artefacts (keeps your data)
	-$(COMPOSE) down
	rm -rf $(APP_DIR)/dist $(APP_DIR)/coverage .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

reset: ## DESTRUCTIVE — also deletes the database and every uploaded document
	@printf 'This deletes the database volume and all uploaded documents. Type "yes": ' \
		&& read ans && [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v

# NexaRAG — common tasks.
#
# `make` on its own prints this list. The one you want on a fresh clone is `make setup`.

APP_DIR := ../Nexarag-app
COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help setup setup-native up down restart logs ps shell db-shell migrate revision seed \
        dev test test-api test-app verify build clean reset

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- getting started --------------------------------------------------------------------

setup: ## Clone the SPA, write .env, build and start everything (Docker)
	@./scripts/bootstrap.sh

setup-native: ## Same, without Docker — needs your own MySQL, Python 3.12 and Node
	@./scripts/bootstrap.sh --native

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

# --- development (native) ---------------------------------------------------------------

dev: ## Run both dev servers with hot reload (Ctrl-C stops both)
	@echo "API → http://localhost:8000   SPA → http://localhost:3000"
	@trap 'kill 0' INT TERM EXIT; \
	( . .venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000 ) & \
	( cd $(APP_DIR) && npm run dev ) & \
	wait

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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

-include env.sh
export


help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
    
pre-commit: ## Runs the pre-commit checks over entire repo
	cd pipelines && \
	poetry run pre-commit run --all-files

setup: ## Set up local environment for Python development on pipelines
	@pip install pip --upgrade && \
	pip install poetry --upgrade && \
	cd pipelines && \
	poetry install --with dev

test-trigger: ## Runs unit tests for the pipeline trigger code
	@cd pipelines && \
	poetry run python -m pytest tests/trigger

compile-pipeline: ## Compile the pipeline to pipeline.yaml. Must specify pipeline=<training|prediction>
	@cd pipelines/src && \
	poetry run kfp dsl compile --py pipelines/${pipeline}/pipeline.py --output pipelines/${pipeline}/pipeline.yaml --function pipeline

setup-components: ## Run unit tests for a component group
	@cd "components/${GROUP}" && \
	poetry install --with dev

setup-all-components: ## Run unit tests for all pipeline components
	@set -e && \
	for component_group in components/*/ ; do \
		echo "Setup components under $$component_group" && \
		$(MAKE) setup-components GROUP=$$(basename $$component_group) ; \
	done

test-components: ## Run unit tests for a component group
	@cd "components/${GROUP}" && \
	poetry run pytest

test-all-components: ## Run unit tests for all pipeline components
	@set -e && \
	for component_group in components/*/ ; do \
		echo "Test components under $$component_group" && \
		$(MAKE) test-components GROUP=$$(basename $$component_group) ; \
	done

test-components-coverage: ## Run tests with coverage
	@cd "components/${GROUP}" && \
	poetry run coverage run -m pytest && \
	poetry run coverage report -m

test-all-components-coverage: ## Run tests with coverage
	@set -e && \
	for component_group in components/*/ ; do \
		echo "Test components under $$component_group" && \
		$(MAKE) test-components-coverage GROUP=$$(basename $$component_group) ; \
	done

run: ## Compile pipeline and run pipeline in sandbox environment. Must specify pipeline=<training|prediction>. Optionally specify enable_pipeline_caching=<true|false> (defaults to default Vertex caching behaviour)
	@ $(MAKE) compile-pipeline && \
	cd pipelines/src && \
	poetry run python -m pipelines.trigger --template_path=pipelines/${pipeline}/pipeline.yaml --enable_caching=$(enable_pipeline_caching)

e2e-tests: ## Perform end-to-end (E2E) pipeline tests. Must specify pipeline=<training|prediction>. Optionally specify enable_pipeline_caching=<true|false> (defaults to default Vertex caching behaviour).
	@ cd pipelines && \
	poetry run pytest --log-cli-level=INFO tests/$(pipeline) --enable_caching=$(enable_pipeline_caching)

env ?= dev
deploy-infra: ## Deploy the Terraform infrastructure to your project. Requires VERTEX_PROJECT_ID and VERTEX_LOCATION env variables to be set in env.sh. Optionally specify env=<dev|test|prod> (default = dev)
	@ cd terraform/envs/$(env) && \
	terraform init -backend-config='bucket=${VERTEX_PROJECT_ID}-tfstate' && \
	terraform apply -var 'project_id=${VERTEX_PROJECT_ID}' -var 'region=${VERTEX_LOCATION}'

destroy-infra: ## DESTROY the Terraform infrastructure in your project. Requires VERTEX_PROJECT_ID and VERTEX_LOCATION env variables to be set in env.sh. Optionally specify env=<dev|test|prod> (default = dev)
	@ cd terraform/envs/$(env) && \
	terraform init -backend-config='bucket=${VERTEX_PROJECT_ID}-tfstate' && \
	terraform destroy -var 'project_id=${VERTEX_PROJECT_ID}' -var 'region=${VERTEX_LOCATION}'

build-training-container: ## Build and push training container image using Docker
	@ cd model && \
	poetry export -f requirements.txt -o training/requirements.txt && \
	cd training && \
	gcloud builds submit . \
	--tag=${TRAINING_CONTAINER_IMAGE} \
	--region=${VERTEX_LOCATION} \
	--project=${VERTEX_PROJECT_ID} \
	--gcs-source-staging-dir=gs://${VERTEX_PROJECT_ID}-staging/source

build-serving-container: ## Build and push serving container image using Docker
	@ cd model && \
	poetry export --with serving -f requirements.txt -o serving/requirements.txt && \
	cd serving && \
	gcloud builds submit . \
	--tag=${SERVING_CONTAINER_IMAGE} \
	--region=${VERTEX_LOCATION} \
	--project=${VERTEX_PROJECT_ID} \
	--gcs-source-staging-dir=gs://${VERTEX_PROJECT_ID}-staging/source

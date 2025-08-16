# Database initialization script
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    pipeline_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    fairness_score FLOAT,
    performance_score FLOAT,
    explainability_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fairness_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES pipeline_runs(run_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_deployments (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    deployment_status VARCHAR(50) NOT NULL,
    fairness_validated BOOLEAN DEFAULT FALSE,
    explainability_validated BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_fairness_metrics_run_id ON fairness_metrics(run_id);
CREATE INDEX idx_model_deployments_model_id ON model_deployments(model_id);

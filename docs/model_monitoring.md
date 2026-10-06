# Model monitoring

## Start services (Docker Desktop on macOS)

Keep the existing FastAPI model serving process running on port 8000 with
`/metrics/` available. Then, from the repository root:

```bash
docker compose -f docker-compose.monitoring.yml config --quiet
docker compose -f docker-compose.monitoring.yml up -d
docker compose -f docker-compose.monitoring.yml ps
```

Prometheus scrapes `host.docker.internal:8000/metrics/` every 5 seconds.
Open http://localhost:9090/targets and confirm `tomato-serving` is UP.
Query `up{job="tomato-serving"}`: the value should be 1.

## Grafana

Open http://localhost:3000 and log in with `admin` / `admin` on the first
startup, then change the password when prompted. The Prometheus data source
and **Tomato Monitoring / Tomato Model Serving** dashboard are provisioned
from files. Grafana connects to `http://prometheus:9090` inside Compose.
Dashboard: http://localhost:3000/d/tomato-serving/tomato-model-serving

Panels show target health, total predictions, request rate, class counts,
p95 and mean inference latency, and error rate. Latency panels need prediction
traffic within the selected time window; idle periods may show no data.
Counters reset when FastAPI restarts. Rates require at least two scrapes.

To generate valid prediction traffic, use the existing serving API's `/docs`
page at http://localhost:8000/docs with a JPEG or PNG image.

## Serving alert

`TomatoServingDown` becomes pending when a scrape fails and firing after one
minute of continuous failure. View http://localhost:9090/alerts. It returns
to inactive after successful scrapes resume. No email/Slack notifications are
configured; external delivery requires Alertmanager and a configured receiver.

## Batch drift and performance monitoring

The existing scripts are preserved:

```bash
python -m pip install -r requirements-monitoring.txt
python scripts/06_prepare_monitoring_data.py
python scripts/07_evidently_drift.py
python scripts/08_retrain_decision.py
```

Run them in the project's Python environment with the MLflow champion model
available. Reference data uses validation images; current data uses test images
as an offline demonstration. `--simulate-drift` on the preparation script
creates darker, blurred current images for a drift demonstration. It overwrites
local monitoring CSVs, so run it only when you intend to regenerate that data.

The HTML report and summary JSON are local files under `reports/`.
The decision script recommends RETRAIN for degraded performance, WATCH for
drift with acceptable performance, or OK. It does not launch retraining.
For automatic retraining and gated local deployment, see `automatic_pipeline.md`
and run `python scripts/09_auto_pipeline.py --mode monitor` after bootstrapping
its managed API. The original decision script remains advisory when used alone.

Batch drift/performance results are not exported to Prometheus by these scripts;
the Grafana dashboard displays serving metrics only.

## Data and Git

Runtime storage uses Docker named volumes, not repository folders. Do not
stage `dataset/`, `mlruns/`, `processed_data/`, `reports/`, or `monitoring_data/`.
Stage individual source/config/documentation paths; review `git diff --cached`
before committing. Existing `.gitignore` rules cover the generated monitoring
reports and data.

```bash
docker compose -f docker-compose.monitoring.yml down
```

This stops services while retaining their named-volume data. Do not use `down -v`
unless you intend to delete the stored Prometheus and Grafana data.

## Troubleshooting

- DOWN target: check FastAPI, `/metrics/`, and access from Docker to the host.
  If necessary, start FastAPI listening on `0.0.0.0:8000`.
- Grafana data source: use `http://prometheus:9090`, not `localhost:9090`.
- Startup errors: inspect `docker compose -f docker-compose.monitoring.yml logs`.
- Ports 3000/9090 are bound to localhost; access the dashboards from this Mac.

Configuration references:
- https://prometheus.io/docs/prometheus/latest/configuration/configuration/
- https://grafana.com/docs/grafana/latest/administration/provisioning/

# Demonstration Guide for TMA 05 Submission

## Required Visual Evidence

### 1. Pipeline Execution Demo
**Show:** Complete pipeline run from start to finish
**Command to run:**
```bash
python scripts/run_pipeline.py --stage all
```
**Screenshot points:**
- Pipeline starting message
- Each stage completion
- Final success message
- Generated outputs in pipeline_outputs/

### 2. Fairness Gates in Action
**Show:** Bias detection and fairness checking
**Evidence location:** 
- Check `pipeline_outputs/latest/pipeline_report.json`
- Show fairness metrics and thresholds
- Demonstrate bias mitigation results

### 3. API Functionality
**Show:** Working FastAPI endpoints
**Commands:**
```bash
# Start API
python -m uvicorn src.devethops.api.app:app --reload

# Test endpoints (in new terminal)
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### 4. Docker Containerization
**Show:** Successful container builds
**Commands:**
```bash
docker build -t devethops-pipeline .
docker images | grep devethops
```

### 5. GitHub Actions CI/CD
**Show:** Automated pipeline in GitHub
**Evidence:**
- Push code to GitHub
- Show Actions tab with running workflow
- Display all checks passing
- Show deployment artifacts

### 6. Monitoring Dashboard
**Show:** Prometheus/Grafana setup
**Commands:**
```bash
docker-compose up -d
# Show Grafana at http://localhost:3000
# Show Prometheus at http://localhost:9090
```

### 7. Test Results
**Show:** 100% test success rate
**Commands:**
```bash
python validate_flow.py
python test_api.py
python test_pipeline.py
```

## Video Structure (5-10 minutes)
1. **Introduction** (30 seconds)
   - Project overview
   - Ethical AI objectives

2. **Pipeline Demo** (2-3 minutes)
   - Run complete pipeline
   - Show stage progression
   - Highlight ethical checkpoints

3. **Results Analysis** (2 minutes)
   - Show generated outputs
   - Explain fairness metrics
   - Model performance review

4. **CI/CD Integration** (2 minutes)
   - GitHub Actions workflow
   - Docker containerization
   - API endpoints testing

5. **Monitoring Setup** (1-2 minutes)
   - Prometheus metrics
   - Grafana dashboards
   - Alert configuration

6. **Conclusion** (30 seconds)
   - Key achievements
   - Production readiness

## Screenshot Checklist
□ Pipeline execution start
□ All stages completing successfully
□ Final pipeline report
□ Fairness metrics dashboard
□ API health check response
□ Docker build success
□ GitHub Actions passing
□ Prometheus metrics page
□ Grafana dashboard
□ Test results (100% pass rate)

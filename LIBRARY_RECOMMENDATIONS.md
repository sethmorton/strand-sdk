# Library Recommendations for Phase 5-6

## 🎯 Goal
Research and recommend the best Python packages for:
1. **CMA-ES Strategy** (continuous optimization)
2. **Constraint Satisfaction** (hard constraints)
3. **Manifests/Logging** (reproducibility & provenance)

---

## 1. CMA-ES Strategy 🔍

### **Top Recommendation: PyCMA**
- **Package**: `pycma`
- **GitHub**: https://github.com/CMA-ES/pycma
- **Documentation**: https://pycma.gforge.inria.fr/
- **PyPI**: https://pypi.org/project/cma/

**Why PyCMA?**
- ✅ Reference implementation of CMA-ES (Covariance Matrix Adaptation)
- ✅ Well-maintained by original CMA-ES authors
- ✅ Extensive documentation and tutorials
- ✅ Active community
- ✅ Supports both continuous and discrete variants

**Installation**:
```bash
pip install cma
```

**Basic Usage**:
```python
import cma
es = cma.CMAEvolutionStrategy([0]*10, 0.3, {'maxfevals': 1000})
while not es.stop():
    X = es.ask()  # ask for solutions
    es.tell(X, [objective(x) for x in X])  # tell fitness values
```

**Integration with Strand SDK**:
- Treat as a continuous optimizer
- Discretize output for sequences
- Wrap in Strategy protocol

### Alternative: Deap
- **Package**: `deap`
- **GitHub**: https://github.com/DEAP/deap
- **Note**: More general evolutionary algorithms library, CMA-ES is just one option

---

## 2. Constraint Satisfaction 🔍

### **Top Recommendation: python-constraint**
- **Package**: `python-constraint`
- **GitHub**: https://github.com/python-constraint/python-constraint
- **PyPI**: https://pypi.org/project/constraint/

**Why python-constraint?**
- ✅ Pure Python, no external dependencies
- ✅ Simple, intuitive API
- ✅ Good for discrete constraint satisfaction
- ✅ Backtracking solver built-in
- ✅ Well-documented

**Installation**:
```bash
pip install constraint
```

**Basic Usage**:
```python
from constraint import Problem

problem = Problem()
problem.addVariables([1, 2, 3], range(1, 10))
problem.addConstraint(lambda a, b, c: a + b < c, [1, 2, 3])
solutions = problem.getSolutions()
```

**Integration with Strand SDK**:
- Use for hard constraint checking
- Validate sequences before/after optimization
- Filter infeasible candidates

### Alternatives:

#### **OR-Tools** (Google)
- **Package**: `ortools`
- **Best for**: Combinatorial optimization, complex constraints
- **More heavyweight** but very powerful

#### **Pyomo** (Sandia National Labs)
- **Package**: `pyomo`
- **Best for**: Mathematical modeling, continuous + discrete
- **More academic**, heavyweight for biological sequences

---

## 3. Manifests/Logging 🔍

### **Top Recommendation: MLflow**
- **Package**: `mlflow`
- **GitHub**: https://github.com/mlflow/mlflow
- **Documentation**: https://mlflow.org/
- **PyPI**: https://pypi.org/project/mlflow/

**Why MLflow?**
- ✅ Industry standard for ML experiment tracking
- ✅ Simple REST API
- ✅ Tracks parameters, metrics, artifacts
- ✅ Model registry
- ✅ Easy reproducibility
- ✅ Web UI built-in

**Installation**:
```bash
pip install mlflow
```

**Basic Usage**:
```python
import mlflow

mlflow.set_experiment("strand-optimization")
with mlflow.start_run():
    mlflow.log_param("strategy", "GA")
    mlflow.log_metric("best_score", 0.95)
    mlflow.log_artifact("results.json")
```

**Integration with Strand SDK**:
```python
import mlflow
from strand.engine import Engine

engine = Engine(...)
with mlflow.start_run():
    mlflow.log_params({
        "strategy": config.method,
        "iterations": config.iterations,
        "population_size": config.population_size,
    })
    
    results = engine.run()
    
    for i, stats in enumerate(results.history):
        mlflow.log_metrics({
            "best": stats.best,
            "mean": stats.mean,
            "std": stats.std,
        }, step=i)
    
    mlflow.log_artifact("manifest.json")
```

### Alternatives:

#### **Weights & Biases** (W&B)
- **Package**: `wandb`
- **Best for**: Team collaboration, beautiful dashboards
- **Cloud-hosted** (MLflow is local/self-hosted)

#### **Neptune.ai**
- **Package**: `neptune-client`
- **Best for**: Team collaboration, advanced visualizations
- **Also cloud-hosted**

#### **Custom JSON Manifest** (Lightweight)
- Just use standard `json.dump()` / `json.load()`
- Store experiment metadata, parameters, results
- **Best for**: Simple reproducibility without external services

---

## 📋 Recommendation Summary

| Feature | Package | Version | Use Case |
|---------|---------|---------|----------|
| **CMA-ES** | `pycma` | 3.x | Continuous optimization, reference implementation |
| **Hard Constraints** | `python-constraint` | 1.4.x | Discrete CSP, validation, filtering |
| **Experiment Tracking** | `mlflow` | 2.x | Full reproducibility, web UI, team ready |

---

## 🚀 Implementation Plan

### Phase 5a: CMA-ES Strategy
1. Add `pycma` to `requirements.txt`
2. Implement `CMAESStrategy` in `strand/engine/strategies/cmaes.py`
   - Inherit from Strategy protocol
   - Wrap PyCMA's `CMAEvolutionStrategy`
   - Discretize continuous output for sequences
3. Write tests in `tests/engine/test_cmaes.py`
4. Update strategy factory in `strand/engine/strategies/__init__.py`

### Phase 5b: Constraint Satisfaction
1. Add `python-constraint` to `requirements.txt`
2. Create `strand/engine/constraints/solver.py`
   - Use for hard constraint validation
   - Filter infeasible candidates
   - Optional: pre-generate feasible set
3. Update `LocalExecutor` and `LocalPoolExecutor` to use constraint checking
4. Write tests in `tests/engine/test_constraint_satisfaction.py`

### Phase 5c: Manifests & Logging (MLflow)
1. Add `mlflow` to `requirements.txt`
2. Create `strand/logging/mlflow_tracker.py`
   - Wrap engine runs
   - Track parameters, metrics, artifacts
   - Handle manifest generation
3. Create example in `examples/engine_with_tracking.py`
4. Write tests in `tests/logging/test_mlflow_tracker.py`

---

## 📦 Updated Requirements

Add to `requirements.txt`:
```
# Phase 5+ packages
cma>=3.0.0              # CMA-ES optimization
constraint>=1.4.0       # Constraint satisfaction
mlflow>=2.0.0           # Experiment tracking
```

---

## 🎓 Quick Reference

### PyCMA
```python
import cma
es = cma.CMAEvolutionStrategy(x0, sigma0)
while not es.stop():
    solutions = es.ask()
    fitness = [evaluate(s) for s in solutions]
    es.tell(solutions, fitness)
```

### python-constraint
```python
from constraint import Problem
p = Problem()
p.addVariable('x', range(10))
p.addConstraint(lambda x: x > 5)
solutions = p.getSolutions()
```

### MLflow
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("key", value)
    mlflow.log_metric("metric", score)
    mlflow.log_artifact("path/to/file")
```

---

## ✅ Decision Matrix

| Library | Maturity | Community | Documentation | Ease of Use | Best For |
|---------|----------|-----------|---|---|---|
| PyCMA | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | CMA-ES |
| python-constraint | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | CSP |
| MLflow | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Tracking |

---

## 🔗 Resources

- **PyCMA**: https://pycma.gforge.inria.fr/
- **python-constraint**: https://github.com/python-constraint/python-constraint
- **MLflow**: https://mlflow.org/docs/latest/

All three are actively maintained, well-documented, and production-ready.


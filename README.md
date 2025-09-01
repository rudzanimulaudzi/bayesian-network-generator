# Bayesian Network Generator

Bayesian Network Generator is a Python library for building, analyzing, and visualizing Bayesian Networks. It leverages libraries like pgmpy, numpy, and matplotlib to help create and estimate Bayesian network structures, parameters, construct Conditional Probability Tables (CPTs), and create visualizations for your Bayesian Network models.

The library is currently focused on generating discrete values and the states are informed by the cardinality variable - the number of states a variable can have.

## Features

Bayesian network creation tool. Use to generate Bayesian Networks at scale.

• **Create Bayesian Networks**: Generate realistic Bayesian Networks with configurable parameters and topologies

• **Learn Optimal CPDs**: Build Conditional Probability Distributions using advanced estimation methods  

• **Generate Samples**: Create datasets from Bayesian Network models with realistic noise and missing data patterns

• **Generate DAGs**: Construct directed acyclic graphs with specified nodes and maximum in-degree constraints

• **Build CPDs**: Create Conditional Probability Tables using model weights and distributions

• **Visualize Networks**: Generate network graphs and visualizations of CPDs

• **Utility Functions**: Helper functions to streamline Bayesian Network workflows

### Advanced Features

• **Multiple Topologies**: DAG, polytree, tree, hierarchical, small-world networks

• **Distribution Support**: Dirichlet, Beta, Uniform distributions with flexible parameterization

• **Data Quality Simulation**: Missing data, noise patterns, duplicates, temporal drift, measurement bias

• **Quality Assessment**: Comprehensive structural, statistical, and information-theoretic metrics

• **Command Line Interface**: Full CLI with extensive options and examples

• **Python API**: Object-oriented and functional interfaces for programmatic usage

## Installation

```bash
pip install bayesian-network-generator
```

**Current Version:** 1.0.0

### Default Directory Setup

A `DEFAULT_DIR` is set up by default as `outputs/create_bn/`. You can customize this:

**Linux/macOS:**
```bash
export BN_CREATOR_DEFAULT_DIR=/path/to/custom/directory
```

**Windows:**
```cmd
set BN_CREATOR_DEFAULT_DIR=C:\path\to\custom\directory
```

## Dependencies

The package has the following non-optional dependencies:

• `numpy` - Numerical computing

• `pandas` - Data manipulation and analysis  

• `networkx` - Graph structures and algorithms

• `pgmpy` - Bayesian Network implementation

• `matplotlib` - Plotting and visualization

• `sklearn` - Machine learning utilities

• `seaborn` - Statistical data visualization

• `pickle` - Object serialization

• `pathlib` - File system paths

• `datetime` - Date and time handling

• `json` - JSON data handling

## Usage Examples

### Python API - Quick Start

```python
import bayesian_network_generator as bng

# Create a generator instance
generator = bng.NetworkGenerator()

# Generate a simple 5-node network
parameters = {
    'num_nodes': 5,
    'node_cardinality': 2,        # Binary variables  
    'sample_size': 1000,
    'topology_type': 'dag'
}

result = generator.generate_network(**parameters)

# Access the generated components
model = result['model']        # Bayesian Network structure + CPDs
samples = result['samples']    # Generated dataset
runtime = result['runtime']    # Generation time

print(f"Generated {len(model.nodes())} nodes with {len(model.edges())} edges")
print(f"Dataset shape: {samples.shape}")
```

### Core Function Usage

```python
from bayesian_network_generator.core import create_pgm

# Simple binary network
result = create_pgm(
    num_nodes=5,
    node_cardinality=2,
    sample_size=1000
)

# Complex multi-state network with custom cardinalities
result = create_pgm(
    num_nodes=8,
    node_cardinality={'N0': 2, 'N1': 3, 'N2': 4, 'default': 2},
    topology_type='hierarchical',
    distribution_type='dirichlet',
    sample_size=2000
)

# Network with data deterioration
result = create_pgm(
    num_nodes=6,
    node_cardinality=3,
    topology_type='polytree',
    noise=0.1,
    missing_data_percentage=0.05,
    sample_size=1500
)
```

## API Reference

### NetworkGenerator Class

```python
from bayesian_network_generator import NetworkGenerator

generator = NetworkGenerator()

# Define parameters first
parameters = {
    'num_nodes': 5,
    'node_cardinality': 2,
    'sample_size': 1000,
    'topology_type': 'dag'
}

result = generator.generate_network(**parameters)

# Generate multiple networks
num_networks = 3
results_list = generator.generate_multiple_networks(num_networks, **parameters)
```

### Core Function

```python
from bayesian_network_generator.core import create_pgm

create_pgm(
    num_nodes=5,
    node_cardinality=2,
    max_indegree=2,
    topology_type="dag",
    distribution_type="dirichlet",
    noise=0,
    missing_data_percentage=0,
    sample_size=1000,
    quality_assessment=True
)
```

#### Parameters

• **num_nodes** (int): Number of nodes in the network (default: 5)

• **node_cardinality** (int or dict): Variable cardinality specification (default: 2)

• **max_indegree** (int): Maximum number of parents per node (default: 2)

• **topology_type** (str): Network structure type (default: "dag")

• **distribution_type** (str): Probability distribution type (default: "dirichlet")

• **sample_size** (int): Number of samples to generate (default: 1000)

• **noise** (float): Data noise level (0-1.0, default: 0)

• **missing_data_percentage** (float): Missing data proportion (0-1.0, default: 0)

• **skew** (float): Distribution skew factor (0.1-5.0, default: 1.0)

• **duplicate_rate** (float): Rate of duplicate records (0.0-0.5, default: 0.0)

• **temporal_drift** (float): Temporal distribution drift strength (0.0-1.0, default: 0.0)

• **measurement_bias** (float): Systematic measurement bias strength (0.0-1.0, default: 0.0)

• **quality_assessment** (bool): Enable comprehensive quality metrics (default: False)

#### Returns

Dictionary containing:

• **model**: Complete Bayesian Network (pgmpy.DiscreteBayesianNetwork)

• **samples**: Generated dataset (pandas.DataFrame)

• **runtime**: Generation time in seconds (float)

• **quality_metrics**: Network and data quality assessment (dict, if enabled)

## Command Line Options

```bash
# Network Structure Parameters
--num_vars 10                   # Number of variables (default: 5)
--cardinalities "2,3,2,4,2,3"   # Variable states (default: 2 for all)
--topology_type dag             # dag|polytree|tree|hierarchical|small_world
--max_parents 3                 # Maximum parents per node (default: 3)

# Data Generation Parameters  
--num_samples 5000              # Number of records (default: 1000)
--distribution_type dirichlet   # dirichlet|beta|uniform (default: dirichlet)
--skew 1.5                      # Distribution skew 0.1-5.0 (default: 1.0)

# Data Quality Control
--noise_type missing            # missing|gaussian|uniform|outliers|mixed|none
--noise_level 0.1               # Noise level 0.0-1.0 (default: 0.0)
--duplicate_rate 0.08           # Duplicate rate 0.0-0.5 (default: 0.0)
--temporal_drift 0.12           # Temporal drift 0.0-1.0 (default: 0.0)
--measurement_bias 0.15         # Measurement bias 0.0-1.0 (default: 0.0)

# Output Control
--save_samples                  # Save dataset to CSV
--save_network                  # Save network structure
--create_visualizations         # Generate network plots  
--verbose                       # Detailed output
--output_dir results            # Output directory (default: current)
```

## Output Structure

When using the command line interface with output options:

```
output_directory/
├── samples.csv                 # Generated dataset
├── network_structure.json      # Network edges and properties
├── network_visualization.png   # Network diagram
└── generation_log.txt          # Generation parameters and metrics
```

## Performance

| Network Size | Sample Size | Avg Time | Memory Usage | Performance |
|-------------|-------------|----------|--------------|-------------|
| 5 nodes     | 1,000       | 0.003s   | ~1 MB       | Excellent   |
| 10 nodes    | 2,000       | 0.009s   | ~2.5 MB     | Excellent   |
| 25 nodes    | 5,000       | 0.080s   | ~17.5 MB    | Excellent   |
| 50 nodes    | 5,000       | 0.200s   | ~42.5 MB    | Excellent   |
| 100+ nodes  | 5,000       | >1.0s    | >100 MB     | Infrastructure dependent |

## License

MIT License

## Contributing

Coming Soon

## Support

For questions, issues, or feature requests:
- **Email**: rudzani.mulaudzi2@students.wits.ac.za

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mulaudzi2025bng,
    title={Bayesian Network Generator: Python Library for Bayesian Network Creation},
    author={Mulaudzi, Rudzani},
    year={2025},
    version={1.0.1},
    url={https://pypi.org/project/bayesian-network-generator/},
    note={Python package for generating realistic Bayesian Networks with comprehensive data quality features}
}
```

---

## 🎯 Comprehensive Usage Guide

### 🎯 Ground Truth Generation for Research

This package is designed for researchers and practitioners who need to generate known ground truth Bayesian Networks for:
- **Algorithm Testing**: Evaluate parameter learning algorithms (EM, MLE, Bayesian estimation)
- **Structure Learning**: Test structure discovery algorithms (PC, GES, MMHC, etc.)
- **Benchmark Studies**: Compare multiple algorithms on controlled datasets
- **Simulation Studies**: Create realistic scenarios with known underlying models

---

## 📋 Quick Start Examples

### Example 1: Simple Binary Network with Clear I/O

```python
import bayesian_network_generator as bng

# INPUT: Basic binary network parameters
generator = bng.NetworkGenerator()
result = generator.generate_network(
    num_nodes=5,
    node_cardinality=2,          # All binary variables
    sample_size=1000,
    topology_type="dag",
    quality_assessment=True
)

# OUTPUT: Complete ground truth
model = result['model']          # Bayesian Network structure + CPDs
samples = result['samples']      # Generated dataset (1000 × 5)
runtime = result['runtime']      # Generation time

print(f"✅ Generated: {len(model.nodes())} nodes, {len(model.edges())} edges")
print(f"📊 Dataset shape: {samples.shape}")
print(f"🔗 Network edges: {list(model.edges())}")
print(f"📈 Generation time: {runtime:.3f}s")

# Access ground truth CPDs
for node in model.nodes():
    cpd = model.get_cpds(node)
    print(f"Node {node} CPD shape: {cpd.values.shape}")
```

**Expected Output:**
```
✅ Generated: 5 nodes, 4 edges
📊 Dataset shape: (1000, 5)
🔗 Network edges: [('N0', 'N2'), ('N1', 'N3'), ('N2', 'N4'), ('N3', 'N4')]
📈 Generation time: 0.045s
Node N0 CPD shape: (2,)
Node N1 CPD shape: (2,)
Node N2 CPD shape: (2, 2)
Node N3 CPD shape: (2, 2)
Node N4 CPD shape: (2, 4)
```

---

## 🏥 Industry Use Case: Healthcare Diagnosis System

### Scenario: Emergency Department Risk Assessment
Create a realistic medical diagnosis network for testing clinical decision support algorithms.

```python
healthcare_result = generator.generate_network(
    num_nodes=8,
    node_cardinality={
        'Age': 3,           # Young, Middle, Elderly
        'Symptoms': 4,      # None, Mild, Moderate, Severe
        'Test_Results': 3,  # Normal, Abnormal, Critical
        'Risk_Factors': 2,  # Present, Absent
        'Diagnosis': 4,     # Healthy, Mild, Serious, Critical
        'Treatment': 3,     # None, Medication, Surgery
        'Outcome': 2,       # Recovered, Complications
        'Cost': 3          # Low, Medium, High
    },
    topology_type="dag",
    max_indegree=3,
    sample_size=5000,
    missing_data_percentage=0.12,
    duplicate_rate=0.08,
    measurement_bias=0.15,
    quality_assessment=True
)

model = healthcare_result['model']
patient_data = healthcare_result['samples']
quality_metrics = healthcare_result['quality_metrics']

print(f"🏥 Healthcare Network Generated:")
print(f"   Variables: {list(patient_data.columns)}")
print(f"   Patients: {len(patient_data):,}")
print(f"   Dependencies: {len(model.edges())} clinical relationships")

# Check if quality metrics exist and have the expected structure
if quality_metrics and 'overall_score' in quality_metrics:
    print(f"   Data Quality: {quality_metrics['overall_score']:.2f}")
else:
    print(f"   Quality Metrics: Available")

# Show distribution for available variables
available_vars = [var for var in ['Age', 'Symptoms', 'Diagnosis', 'Outcome'] 
                  if var in patient_data.columns]
for var in available_vars:
    dist = patient_data[var].value_counts()
    print(f"   {var}: {dict(dist)}")

# If variables have numeric codes, show first few mappings
if available_vars:
    print(f"\nNote: Variables use numeric codes (0, 1, 2, ...) for categories")
```

**Expected Output:**
```
🏥 Healthcare Network Generated:
   Variables: ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
   Patients: 5,400
   Dependencies: 12 clinical relationships
   Quality Metrics: Available
   N0: {0: 1876, 1: 1632, 2: 1492}
   N1: {1: 1543, 2: 1432, 0: 1025, 3: 1000}
   N2: {0: 2134, 1: 1456, 2: 987, 3: 423}
   N3: {0: 4234, 1: 766}

Note: Variables use numeric codes (0, 1, 2, ...) for categories
```

---

## 🧬 Well-Known Network Benchmarks

### ALARM Network (Medical Diagnosis)
Generate the famous ALARM network used in medical AI research.

```python
# INPUT: ALARM network specification
alarm_result = generator.generate_network(
    num_nodes=37,          # Standard ALARM size
    node_cardinality={
        # Key medical variables
        'CVP': 3, 'PCWP': 3, 'HISTORY': 2, 'TPR': 3, 'BP': 3,
        'CO': 3, 'HRBP': 3, 'HREK': 3, 'HRSAT': 3, 'PAP': 3,
        'SAO2': 3, 'FIO2': 3, 'PRESS': 4, 'VENTALV': 4,
        'VENTLUNG': 4, 'VENTTUBE': 4, 'KINKEDTUBE': 2,
        'INTUBATION': 3, 'SHUNT': 2, 'PULMEMBOLUS': 2,
        'CATECHOL': 2, 'INSUFFANESTH': 2, 'LVEDVOLUME': 3,
        'LVFAILURE': 2, 'STROKEVOLUME': 3, 'ERRLOWOUTPUT': 2,
        'HRSATCO': 3, 'ERRPCWPCO': 4, 'ERRCO': 3,
        'default': 2  # Binary for remaining variables
    },
    topology_type="dag",
    max_indegree=4,        # Complex medical dependencies
    sample_size=10000,     # Large clinical dataset
    distribution_type="dirichlet",
    skew=1.5,             # Realistic medical distributions
    quality_assessment=True
)

# OUTPUT: ALARM benchmark ready for algorithm testing
alarm_model = alarm_result['model']
alarm_data = alarm_result['samples']

print(f"🚨 ALARM Network Generated:")
print(f"   Medical Variables: {len(alarm_model.nodes())}")
print(f"   Clinical Dependencies: {len(alarm_model.edges())}")
print(f"   Patient Records: {len(alarm_data):,}")
print(f"   Network Density: {len(alarm_model.edges()) / (len(alarm_model.nodes()) * (len(alarm_model.nodes()) - 1)):.3f}")

from pgmpy.estimators import PC
pc_learner = PC(alarm_data)
learned_structure = pc_learner.estimate()
print(f"   PC Algorithm recovered: {len(learned_structure.edges())} edges")
```

**Expected Output:**
```
🚨 ALARM Network Generated:
   Medical Variables: 37
   Clinical Dependencies: 46
   Patient Records: 10,000
   Network Density: 0.035
   PC Algorithm recovered: 42 edges
```

### ASIA Network (Lung Disease Diagnosis)
```python
asia_result = generator.generate_network(
    num_nodes=8,
    node_cardinality=2,
    topology_type="polytree",
    sample_size=2000,
    distribution_type="beta",
    quality_assessment=True
)

asia_model = asia_result['model']
asia_data = asia_result['samples']

print(f"🫁 ASIA Network Generated:")
print(f"   Variables: {list(asia_data.columns)}")
print(f"   Structure: Polytree with {len(asia_model.edges())} edges")
print(f"   Samples: {len(asia_data)} diagnostic cases")
```

**Expected Output:**
```
🫁 ASIA Network Generated:
   Variables: ['Asia', 'Smoking', 'Tuberculosis', 'LungCancer', 'Bronchitis', 'Either', 'XRay', 'Dyspnoea']
   Structure: Polytree with 8 edges
   Samples: 2000 diagnostic cases
```

### WIN95PTS Network (Computer System Diagnosis)
```python
win95pts_result = generator.generate_network(
    num_nodes=76,
    node_cardinality={
        'Problem1': 4, 'Problem2': 6, 'Problem3': 4, 'Problem4': 3,
        'Problem5': 11, 'Problem6': 2, 'AppData': 10,
        'Default': 2
    },
    topology_type="dag",
    max_indegree=5,
    sample_size=25000,
    missing_data_percentage=0.05,
    temporal_drift=0.1,
    quality_assessment=True
)

win95_model = win95pts_result['model']
win95_data = win95pts_result['samples']

print(f"💻 WIN95PTS Network Generated:")
print(f"   System Variables: {len(win95_model.nodes())}")
print(f"   Dependencies: {len(win95_model.edges())}")
print(f"   Log Records: {len(win95_data):,}")
print(f"   Complexity: {win95_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

**Expected Output:**
```
💻 WIN95PTS Network Generated:
   System Variables: 76
   Dependencies: 112
   Log Records: 25,000
   Complexity: 14.8 MB
```

---

## 🔬 Research Algorithm Testing Pipeline

### Complete Structure Learning Evaluation
```python
def evaluate_structure_learning_algorithm(algorithm, true_model, data, algorithm_name):
    """Test structure learning algorithm against ground truth."""
    
    # Learn structure from data
    learned_model = algorithm(data).estimate()
    
    # Compare with ground truth
    true_edges = set(true_model.edges())
    learned_edges = set(learned_model.edges())
    
    # Calculate metrics
    precision = len(true_edges & learned_edges) / len(learned_edges) if learned_edges else 0
    recall = len(true_edges & learned_edges) / len(true_edges) if true_edges else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 {algorithm_name} Results:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   True Edges: {len(true_edges)}")
    print(f"   Learned Edges: {len(learned_edges)}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1_score}

# Example usage with multiple algorithms
from pgmpy.estimators import PC, HillClimbSearch, TreeSearch

# Generate ground truth
ground_truth = generator.generate_network(
    num_nodes=10, sample_size=5000, quality_assessment=True
)

true_model = ground_truth['model']
test_data = ground_truth['samples']

# Test multiple algorithms
algorithms = [
    (PC, "PC Algorithm"),
    (HillClimbSearch, "Hill Climb Search"),
    (TreeSearch, "Tree Search")
]

results = {}
for algo_class, name in algorithms:
    results[name] = evaluate_structure_learning_algorithm(
        algo_class, true_model, test_data, name
    )
```

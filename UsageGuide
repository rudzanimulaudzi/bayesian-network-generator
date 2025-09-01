# 🚀 Complete Workflow Guide: Bayesian Network Generator

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [Installation & Setup](#installation--setup)
3. [Basic Workflow](#basic-workflow)
4. [Advanced Workflows](#advanced-workflows)
5. [CLI Command Reference](#cli-command-reference)
6. [Function API Reference](#function-api-reference)
7. [Real-World Examples](#real-world-examples)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Quick Start Guide

### 5-Minute Setup & First Network

```bash
# 1. Install the package
pip install bayesian-network-generator

# 2. Create your first network (CLI)
bng-create --nodes 5 --edges 8 --output my_first_network.json

# 3. Generate data from the network
bng-generate --network my_first_network.json --samples 1000 --output data.csv

# 4. Visualize your network
python -c "
from bayesian_network_generator import create_pgm
network, data = create_pgm(nodes=5, edges=8, samples=1000)
print('✅ Network created with', len(network.nodes()), 'nodes')
print('✅ Generated', len(data), 'samples')
"
```

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version  # Should be 3.8 or higher

# Recommended: Create virtual environment
python -m venv bng_env
source bng_env/bin/activate  # On Windows: bng_env\Scripts\activate
```

### Installation Options

#### Option 1: PyPI (Recommended)
```bash
pip install bayesian-network-generator
```

#### Option 2: Development Installation
```bash
git clone https://github.com/rudzanimulaudzi/bayesian-network-generator.git
cd bayesian-network-generator
pip install -e .
```

#### Option 3: With Optional Dependencies
```bash
# For visualization support
pip install bayesian-network-generator[viz]

# For all features including CLI
pip install bayesian-network-generator[full]
```

### Verify Installation
```python
import bayesian_network_generator as bng
print(f"✅ BNG Version: {bng.__version__}")

# Test basic functionality
network, data = bng.create_pgm(nodes=3, edges=2, samples=100)
print(f"✅ Created network with {len(network.nodes())} nodes")
print(f"✅ Generated {len(data)} data samples")
```

---

## Basic Workflow

### Workflow 1: Simple Network Creation

```python
from bayesian_network_generator import create_pgm
import pandas as pd

# Step 1: Create a simple network
network, data = create_pgm(
    nodes=4,           # Number of nodes
    edges=5,           # Number of edges  
    samples=1000,      # Data samples to generate
    node_names=['A', 'B', 'C', 'D']  # Custom node names
)

# Step 2: Inspect the network
print("Network Nodes:", network.nodes())
print("Network Edges:", network.edges())

# Step 3: Examine the data
print("\nData Info:")
print(data.head())
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
```

### Workflow 2: Using CLI for Batch Processing

```bash
# Create multiple networks with different configurations
mkdir networks_batch

# Small networks (research scenarios)
bng-create --nodes 5 --edges 7 --output networks_batch/small_net.json
bng-create --nodes 8 --edges 12 --output networks_batch/medium_net.json
bng-create --nodes 15 --edges 25 --output networks_batch/large_net.json

# Generate datasets from each network
for net in networks_batch/*.json; do
    name=$(basename "$net" .json)
    bng-generate --network "$net" --samples 2000 --output "networks_batch/${name}_data.csv"
done

# View results
ls -la networks_batch/
```

### Workflow 3: Comprehensive Network with Quality Metrics

```python
from bayesian_network_generator import create_comprehensive_pgm, NetworkQualityMetrics

# Step 1: Create comprehensive network with quality issues
results = create_comprehensive_pgm(
    nodes=6,
    edges=10,
    samples=1500,
    missing_data_rate=0.05,    # 5% missing values
    noise_level=0.02,          # 2% noise
    outlier_rate=0.01,         # 1% outliers
    node_names=['Gene1', 'Gene2', 'Protein1', 'Protein2', 'Disease', 'Treatment']
)

network = results['network']
clean_data = results['clean_data']
noisy_data = results['noisy_data']

# Step 2: Analyze data quality
quality_metrics = NetworkQualityMetrics(network, clean_data, noisy_data)
quality_report = quality_metrics.generate_comprehensive_report()

print("📊 Quality Assessment:")
for metric, value in quality_report.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")
```

---

## Advanced Workflows

### Workflow 4: Custom Network Architectures

```python
from bayesian_network_generator import NetworkGenerator
import networkx as nx

# Create custom generator
generator = NetworkGenerator()

# Method 1: Predefined topologies
networks = {
    'chain': generator.create_chain_network(['Start', 'Middle1', 'Middle2', 'End']),
    'star': generator.create_star_network(['Center', 'Node1', 'Node2', 'Node3', 'Node4']),
    'tree': generator.create_tree_network(['Root', 'Branch1', 'Branch2', 'Leaf1', 'Leaf2', 'Leaf3'])
}

for topology, network in networks.items():
    print(f"\n{topology.upper()} Network:")
    print(f"  Nodes: {len(network.nodes())}")
    print(f"  Edges: {len(network.edges())}")
    print(f"  Edges: {list(network.edges())}")

# Method 2: Custom topology from adjacency matrix
import numpy as np

# Define custom structure (6x6 adjacency matrix)
adjacency = np.array([
    [0, 1, 1, 0, 0, 0],  # Node 0 -> Node 1, 2
    [0, 0, 0, 1, 1, 0],  # Node 1 -> Node 3, 4  
    [0, 0, 0, 0, 1, 1],  # Node 2 -> Node 4, 5
    [0, 0, 0, 0, 0, 1],  # Node 3 -> Node 5
    [0, 0, 0, 0, 0, 1],  # Node 4 -> Node 5
    [0, 0, 0, 0, 0, 0]   # Node 5 (sink)
])

custom_network = generator.create_from_adjacency(
    adjacency_matrix=adjacency,
    node_names=['Input', 'Process1', 'Process2', 'Decision', 'Action', 'Output']
)

# Generate data with custom parameters
data = generator.generate_data(
    network=custom_network,
    samples=2000,
    distribution_type='mixed',  # Mix of distributions
    seed=42
)

print(f"\nCustom Network Generated:")
print(f"  Data shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")
```

### Workflow 5: Benchmark Networks (ALARM, ASIA, WIN95PTS)

```python
from bayesian_network_generator import NetworkGenerator

generator = NetworkGenerator()

# Load standard benchmark networks
benchmarks = ['alarm', 'asia', 'win95pts']

for benchmark in benchmarks:
    print(f"\n🔬 Loading {benchmark.upper()} benchmark:")
    
    try:
        # Load the benchmark network
        network = generator.load_benchmark(benchmark)
        
        print(f"  ✅ Nodes: {len(network.nodes())}")
        print(f"  ✅ Edges: {len(network.edges())}")
        
        # Generate data from benchmark
        data = generator.generate_data(network, samples=1000)
        print(f"  ✅ Generated data: {data.shape}")
        
        # Save for later use
        import json
        from pgmpy.readwrite import BIFWriter
        
        # Save network structure
        writer = BIFWriter(network)
        writer.write_bif(f'{benchmark}_network.bif')
        
        # Save data
        data.to_csv(f'{benchmark}_data.csv', index=False)
        
        print(f"  ✅ Saved: {benchmark}_network.bif, {benchmark}_data.csv")
        
    except Exception as e:
        print(f"  ❌ Error loading {benchmark}: {e}")
```

### Workflow 6: Data Quality Analysis & Cleaning

```python
from bayesian_network_generator import NetworkQualityMetrics, create_comprehensive_pgm
import matplotlib.pyplot as plt
import seaborn as sns

# Create network with various quality issues
results = create_comprehensive_pgm(
    nodes=8,
    edges=15,
    samples=2000,
    missing_data_rate=0.10,
    noise_level=0.05, 
    outlier_rate=0.02,
    inconsistency_rate=0.03
)

# Initialize quality analyzer
quality_analyzer = NetworkQualityMetrics(
    network=results['network'],
    original_data=results['clean_data'],
    modified_data=results['noisy_data']
)

# Comprehensive quality analysis
print("🔍 COMPREHENSIVE QUALITY ANALYSIS")
print("=" * 50)

# 1. Missing data analysis
missing_analysis = quality_analyzer.analyze_missing_data()
print(f"\n📊 Missing Data Analysis:")
print(f"  Total missing values: {missing_analysis['total_missing']}")
print(f"  Missing percentage: {missing_analysis['missing_percentage']:.2%}")
print(f"  Columns with missing data: {missing_analysis['columns_with_missing']}")

# 2. Outlier detection
outlier_analysis = quality_analyzer.detect_outliers()
print(f"\n🎯 Outlier Analysis:")
print(f"  Total outliers detected: {outlier_analysis['total_outliers']}")
print(f"  Outlier rate: {outlier_analysis['outlier_rate']:.2%}")

# 3. Data consistency check
consistency_analysis = quality_analyzer.check_consistency()
print(f"\n✅ Consistency Analysis:")
print(f"  Consistency score: {consistency_analysis['consistency_score']:.4f}")
print(f"  Inconsistent rows: {consistency_analysis['inconsistent_rows']}")

# 4. Generate cleaning recommendations
recommendations = quality_analyzer.get_cleaning_recommendations()
print(f"\n🔧 Cleaning Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# 5. Apply automated cleaning
cleaned_data = quality_analyzer.apply_automated_cleaning()
print(f"\n🧹 After Automated Cleaning:")
print(f"  Original shape: {results['noisy_data'].shape}")
print(f"  Cleaned shape: {cleaned_data.shape}")

# 6. Quality improvement metrics
improvement = quality_analyzer.calculate_improvement_metrics(cleaned_data)
print(f"\n📈 Improvement Metrics:")
for metric, value in improvement.items():
    print(f"  {metric}: {value:.4f}")
```

---

## CLI Command Reference

### Complete CLI Usage Guide

#### 1. Network Creation Commands

```bash
# Basic network creation
bng-create --nodes 6 --edges 10 --output network.json

# Advanced network creation with custom parameters
bng-create \
    --nodes 8 \
    --edges 15 \
    --output advanced_network.json \
    --topology "random" \
    --seed 42 \
    --node-names "A,B,C,D,E,F,G,H"

# Create network with specific topology
bng-create \
    --nodes 5 \
    --topology "chain" \
    --output chain_network.json \
    --node-names "Start,Process1,Process2,Process3,End"

# Batch creation with different topologies
for topology in chain star tree random; do
    bng-create --nodes 6 --topology $topology --output ${topology}_network.json
done
```

#### 2. Data Generation Commands

```bash
# Basic data generation
bng-generate \
    --network network.json \
    --samples 1000 \
    --output data.csv

# Advanced data generation with quality parameters
bng-generate \
    --network network.json \
    --samples 2000 \
    --output noisy_data.csv \
    --missing-rate 0.05 \
    --noise-level 0.02 \
    --outlier-rate 0.01 \
    --seed 123

# Generate multiple datasets from same network
for samples in 500 1000 2000 5000; do
    bng-generate \
        --network network.json \
        --samples $samples \
        --output data_${samples}.csv
done
```

#### 3. Benchmark Network Commands

```bash
# Load and generate data from benchmark networks
for benchmark in alarm asia win95pts; do
    echo "Processing $benchmark benchmark..."
    
    # Generate data from benchmark
    bng-benchmark \
        --network $benchmark \
        --samples 1000 \
        --output ${benchmark}_data.csv
    
    # Add quality issues to benchmark data
    bng-benchmark \
        --network $benchmark \
        --samples 1000 \
        --output ${benchmark}_noisy.csv \
        --missing-rate 0.03 \
        --noise-level 0.01
done
```

#### 4. Quality Analysis Commands

```bash
# Analyze data quality
bng-analyze \
    --data data.csv \
    --network network.json \
    --output quality_report.json

# Compare original vs modified data quality
bng-analyze \
    --original clean_data.csv \
    --modified noisy_data.csv \
    --network network.json \
    --output comparison_report.json \
    --visualize

# Batch quality analysis
for file in data/*.csv; do
    name=$(basename "$file" .csv)
    bng-analyze \
        --data "$file" \
        --network network.json \
        --output "reports/${name}_quality.json"
done
```

#### 5. Visualization Commands

```bash
# Create network visualization
bng-visualize \
    --network network.json \
    --output network_plot.png \
    --layout circular \
    --node-size 1000 \
    --font-size 12

# Create data quality visualization
bng-visualize \
    --data data.csv \
    --network network.json \
    --type quality \
    --output quality_plots/ \
    --format png

# Create comprehensive visualization report
bng-visualize \
    --network network.json \
    --data data.csv \
    --type comprehensive \
    --output report.html \
    --interactive
```

---

## Function API Reference

### Core Functions

#### `create_pgm()`
```python
from bayesian_network_generator import create_pgm

# Full parameter specification
network, data = create_pgm(
    nodes=6,                           # Number of nodes
    edges=10,                          # Number of edges
    samples=1000,                      # Data samples
    node_names=None,                   # Custom node names
    edge_probability=0.3,              # Probability of edge creation
    seed=None,                         # Random seed
    distribution_type='categorical',   # Data distribution
    categories_per_node=3,             # Categories for categorical data
    return_structure=False             # Return network structure info
)

# Examples with different configurations
examples = [
    # Small network for testing
    create_pgm(nodes=3, edges=2, samples=100),
    
    # Medium network with custom names
    create_pgm(
        nodes=5, 
        edges=8, 
        samples=500,
        node_names=['Input', 'Process1', 'Process2', 'Decision', 'Output']
    ),
    
    # Large network with mixed distributions
    create_pgm(
        nodes=10, 
        edges=20, 
        samples=2000,
        distribution_type='mixed',
        seed=42
    )
]

for i, (net, data) in enumerate(examples):
    print(f"Example {i+1}: {len(net.nodes())} nodes, {data.shape[0]} samples")
```

#### `create_comprehensive_pgm()`
```python
from bayesian_network_generator import create_comprehensive_pgm

# Comprehensive network with quality control
results = create_comprehensive_pgm(
    nodes=8,                    # Network size
    edges=15,                   # Connectivity
    samples=1500,               # Data size
    missing_data_rate=0.05,     # Missing value rate
    noise_level=0.02,           # Noise level
    outlier_rate=0.01,          # Outlier rate
    inconsistency_rate=0.005,   # Inconsistency rate
    seed=42,                    # Reproducibility
    quality_metrics=True        # Include quality metrics
)

# Access different components
network = results['network']              # The Bayesian network
clean_data = results['clean_data']        # Original clean data
noisy_data = results['noisy_data']        # Data with quality issues
metrics = results['quality_metrics']     # Quality assessment
structure_info = results['structure']     # Network structure details

print("Network Structure:")
print(f"  Nodes: {len(network.nodes())}")
print(f"  Edges: {len(network.edges())}")
print(f"  Clean data shape: {clean_data.shape}")
print(f"  Noisy data shape: {noisy_data.shape}")
print(f"  Quality score: {metrics['overall_quality']:.4f}")
```

### NetworkGenerator Class

```python
from bayesian_network_generator import NetworkGenerator

# Initialize generator
generator = NetworkGenerator(seed=42)

# Method 1: Create random network
random_net = generator.create_random_network(
    nodes=6,
    edges=10,
    node_names=['A', 'B', 'C', 'D', 'E', 'F']
)

# Method 2: Create specific topology
topologies = {
    'chain': generator.create_chain_network(['Start', 'Mid1', 'Mid2', 'End']),
    'star': generator.create_star_network(['Center', 'Spoke1', 'Spoke2', 'Spoke3']),
    'tree': generator.create_tree_network(['Root', 'Branch1', 'Branch2', 'Leaf1', 'Leaf2'])
}

# Method 3: Custom structure
import numpy as np
adj_matrix = np.array([
    [0, 1, 0],
    [0, 0, 1], 
    [0, 0, 0]
])
custom_net = generator.create_from_adjacency(adj_matrix, ['X', 'Y', 'Z'])

# Generate data with different parameters
for name, network in topologies.items():
    data = generator.generate_data(
        network=network,
        samples=1000,
        distribution_type='categorical',
        categories_per_node=2,
        seed=42
    )
    print(f"{name} network data shape: {data.shape}")

# Advanced data generation with quality control
quality_data = generator.generate_data_with_quality_issues(
    network=random_net,
    samples=2000,
    missing_rate=0.03,
    noise_level=0.01,
    outlier_rate=0.005
)

print(f"Quality-controlled data shape: {quality_data.shape}")
```

### NetworkQualityMetrics Class

```python
from bayesian_network_generator import NetworkQualityMetrics, create_comprehensive_pgm

# Create test data
results = create_comprehensive_pgm(nodes=6, edges=10, samples=1000, 
                                 missing_data_rate=0.05, noise_level=0.02)

# Initialize quality analyzer
quality = NetworkQualityMetrics(
    network=results['network'],
    original_data=results['clean_data'],
    modified_data=results['noisy_data']
)

# Comprehensive analysis methods
analyses = {
    'missing_data': quality.analyze_missing_data(),
    'outliers': quality.detect_outliers(), 
    'consistency': quality.check_consistency(),
    'distribution': quality.analyze_distributions(),
    'correlation': quality.analyze_correlations()
}

print("🔍 QUALITY ANALYSIS RESULTS:")
print("=" * 40)

for analysis_type, results in analyses.items():
    print(f"\n📊 {analysis_type.upper()}:")
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  Results: {results}")

# Generate comprehensive report
report = quality.generate_comprehensive_report()

# Export results
quality.export_quality_report(
    filename='quality_analysis.json',
    include_visualizations=True
)

print("\n✅ Quality analysis complete!")
print("📄 Report saved to: quality_analysis.json")
```

---

## Real-World Examples

### Example 1: Medical Diagnosis Network

```python
from bayesian_network_generator import NetworkGenerator, NetworkQualityMetrics
import pandas as pd

# Medical diagnosis scenario
print("🏥 MEDICAL DIAGNOSIS NETWORK")
print("=" * 40)

# Define medical network structure
medical_nodes = [
    'Age', 'Gender', 'Smoking', 'Genetics',
    'Symptoms', 'Test_Result', 'Diagnosis', 'Treatment'
]

generator = NetworkGenerator(seed=42)

# Create medical network with logical dependencies
medical_network = generator.create_chain_network(medical_nodes)

# Add additional dependencies for realism
additional_edges = [
    ('Age', 'Diagnosis'),
    ('Gender', 'Symptoms'),
    ('Smoking', 'Test_Result'),
    ('Genetics', 'Diagnosis')
]

for parent, child in additional_edges:
    medical_network.add_edge(parent, child)

print(f"Created medical network:")
print(f"  Nodes: {len(medical_network.nodes())}")
print(f"  Edges: {len(medical_network.edges())}")
print(f"  Network structure: {list(medical_network.edges())}")

# Generate realistic medical data
medical_data = generator.generate_data(
    network=medical_network,
    samples=5000,
    distribution_type='categorical',
    categories_per_node=3,  # e.g., Low/Medium/High
    seed=42
)

# Add realistic quality issues (common in medical data)
noisy_medical_data = generator.generate_data_with_quality_issues(
    network=medical_network,
    samples=5000,
    missing_rate=0.08,      # 8% missing (common in medical records)
    noise_level=0.03,       # 3% noise (measurement errors)
    outlier_rate=0.02,      # 2% outliers (rare conditions)
    seed=42
)

# Analyze medical data quality
medical_quality = NetworkQualityMetrics(
    network=medical_network,
    original_data=medical_data,
    modified_data=noisy_medical_data
)

quality_report = medical_quality.generate_comprehensive_report()

print(f"\n📊 Medical Data Quality Assessment:")
print(f"  Data completeness: {quality_report['completeness_score']:.3f}")
print(f"  Data consistency: {quality_report['consistency_score']:.3f}")
print(f"  Overall quality: {quality_report['overall_quality_score']:.3f}")

# Save medical datasets
medical_data.to_csv('medical_clean_data.csv', index=False)
noisy_medical_data.to_csv('medical_realistic_data.csv', index=False)

print(f"\n✅ Medical datasets saved:")
print(f"  Clean data: medical_clean_data.csv ({medical_data.shape})")
print(f"  Realistic data: medical_realistic_data.csv ({noisy_medical_data.shape})")
```

### Example 2: Financial Risk Assessment Network

```python
print("\n💰 FINANCIAL RISK ASSESSMENT NETWORK")
print("=" * 45)

# Financial risk factors
financial_nodes = [
    'Credit_Score', 'Income', 'Debt_Ratio', 'Employment_History',
    'Market_Conditions', 'Risk_Assessment', 'Loan_Approval', 'Interest_Rate'
]

# Create financial network
financial_network = generator.create_random_network(
    nodes=len(financial_nodes),
    edges=12,
    node_names=financial_nodes
)

# Generate financial data with different quality scenarios
scenarios = {
    'high_quality': {'missing': 0.01, 'noise': 0.005, 'outliers': 0.001},
    'medium_quality': {'missing': 0.05, 'noise': 0.02, 'outliers': 0.01},
    'low_quality': {'missing': 0.15, 'noise': 0.05, 'outliers': 0.03}
}

financial_datasets = {}

for scenario_name, params in scenarios.items():
    data = generator.generate_data_with_quality_issues(
        network=financial_network,
        samples=3000,
        missing_rate=params['missing'],
        noise_level=params['noise'],
        outlier_rate=params['outliers'],
        seed=42
    )
    
    financial_datasets[scenario_name] = data
    
    # Quick quality assessment
    quality = NetworkQualityMetrics(
        network=financial_network,
        original_data=None,  # No original for comparison
        modified_data=data
    )
    
    missing_analysis = quality.analyze_missing_data()
    outlier_analysis = quality.detect_outliers()
    
    print(f"\n📈 {scenario_name.upper()} Scenario:")
    print(f"  Missing data: {missing_analysis['missing_percentage']:.1%}")
    print(f"  Outliers detected: {outlier_analysis['outlier_rate']:.1%}")
    print(f"  Data shape: {data.shape}")
    
    # Save scenario data
    data.to_csv(f'financial_{scenario_name}_data.csv', index=False)

print(f"\n✅ Financial scenarios generated and saved!")
```

### Example 3: Research Benchmark Comparison

```python
print("\n🔬 RESEARCH BENCHMARK COMPARISON")
print("=" * 40)

# Compare different benchmark networks
benchmarks = ['alarm', 'asia', 'win95pts']
benchmark_results = {}

for benchmark in benchmarks:
    try:
        print(f"\n🔍 Processing {benchmark.upper()} benchmark:")
        
        # Load benchmark network
        network = generator.load_benchmark(benchmark)
        
        # Generate multiple datasets with different quality levels
        datasets = {}
        for quality_level in ['perfect', 'realistic', 'poor']:
            if quality_level == 'perfect':
                data = generator.generate_data(network, samples=2000, seed=42)
            elif quality_level == 'realistic': 
                data = generator.generate_data_with_quality_issues(
                    network, samples=2000, missing_rate=0.03, 
                    noise_level=0.01, outlier_rate=0.005, seed=42
                )
            else:  # poor
                data = generator.generate_data_with_quality_issues(
                    network, samples=2000, missing_rate=0.10,
                    noise_level=0.05, outlier_rate=0.02, seed=42
                )
            
            datasets[quality_level] = data
        
        # Analyze each dataset
        quality_scores = {}
        for quality_level, data in datasets.items():
            if quality_level == 'perfect':
                # Use perfect data as reference for others
                reference_data = data
                continue
            
            quality = NetworkQualityMetrics(network, reference_data, data)
            report = quality.generate_comprehensive_report()
            quality_scores[quality_level] = report['overall_quality_score']
            
            # Save datasets
            data.to_csv(f'{benchmark}_{quality_level}_data.csv', index=False)
        
        benchmark_results[benchmark] = {
            'network_info': {
                'nodes': len(network.nodes()),
                'edges': len(network.edges())
            },
            'quality_scores': quality_scores,
            'datasets': datasets
        }
        
        print(f"  ✅ Nodes: {len(network.nodes())}, Edges: {len(network.edges())}")
        print(f"  ✅ Generated {len(datasets)} quality variants")
        
    except Exception as e:
        print(f"  ❌ Error with {benchmark}: {e}")

# Summary comparison
print(f"\n📊 BENCHMARK COMPARISON SUMMARY:")
print("=" * 50)

for benchmark, results in benchmark_results.items():
    print(f"\n{benchmark.upper()}:")
    print(f"  Network size: {results['network_info']['nodes']} nodes, {results['network_info']['edges']} edges")
    print(f"  Quality scores:")
    for quality_level, score in results['quality_scores'].items():
        print(f"    {quality_level}: {score:.4f}")

print(f"\n✅ Benchmark comparison complete!")
print(f"📁 Generated {len(benchmarks) * 3} dataset files")
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Installation Issues

```bash
# Issue: Permission denied during installation
# Solution: Use user installation
pip install --user bayesian-network-generator

# Issue: Conflicting dependencies
# Solution: Create clean environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install bayesian-network-generator

# Issue: Missing C++ compiler for pgmpy
# Solution: Install build tools
# On macOS:
xcode-select --install
# On Ubuntu/Debian:
sudo apt-get install build-essential
# On Windows:
# Install Visual Studio Build Tools
```

#### 2. Import Errors

```python
# Issue: Cannot import modules
try:
    from bayesian_network_generator import create_pgm
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Solutions:")
    print("1. Reinstall: pip uninstall bayesian-network-generator && pip install bayesian-network-generator")
    print("2. Check Python path: python -c 'import sys; print(sys.path)'")
    print("3. Verify installation: pip list | grep bayesian")

# Issue: Partial imports work
try:
    from bayesian_network_generator import create_pgm
    from bayesian_network_generator import NetworkGenerator  # Might fail
except ImportError as e:
    print(f"Partial import issue: {e}")
    # Fallback approach
    import bayesian_network_generator as bng
    create_pgm = bng.create_pgm
    NetworkGenerator = bng.NetworkGenerator
```

#### 3. Memory Issues with Large Networks

```python
# Issue: Out of memory with large networks
def create_large_network_safely(nodes, edges, samples):
    """Create large networks with memory management."""
    
    # Check available memory
    import psutil
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    
    # Estimate memory requirements
    estimated_memory = (nodes * edges * samples) / 1e6  # Rough estimate in GB
    
    if estimated_memory > available_memory * 0.8:
        print(f"⚠️  Warning: Estimated memory ({estimated_memory:.2f}GB) exceeds available ({available_memory:.2f}GB)")
        
        # Reduce samples or use chunked processing
        max_samples = int(samples * 0.5)
        print(f"🔧 Reducing samples from {samples} to {max_samples}")
        samples = max_samples
    
    # Create network in chunks if still too large
    if samples > 10000:
        print("🔄 Using chunked processing...")
        chunk_size = 5000
        all_data = []
        
        for i in range(0, samples, chunk_size):
            current_chunk = min(chunk_size, samples - i)
            network, data = create_pgm(nodes=nodes, edges=edges, samples=current_chunk)
            all_data.append(data)
            print(f"  Processed chunk {i//chunk_size + 1}: {current_chunk} samples")
        
        # Combine chunks
        import pandas as pd
        combined_data = pd.concat(all_data, ignore_index=True)
        return network, combined_data
    
    else:
        return create_pgm(nodes=nodes, edges=edges, samples=samples)

# Example usage
network, data = create_large_network_safely(nodes=20, edges=50, samples=50000)
print(f"✅ Successfully created large network: {data.shape}")
```

#### 4. Data Quality Issues

```python
# Issue: Unexpected data quality results
def debug_data_quality(network, data):
    """Debug data quality issues step by step."""
    
    print("🔍 DEBUGGING DATA QUALITY")
    print("=" * 30)
    
    # Basic data info
    print(f"Data shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Check for obvious issues
    print(f"\n📊 Basic Quality Checks:")
    print(f"  Null values: {data.isnull().sum().sum()}")
    print(f"  Duplicate rows: {data.duplicated().sum()}")
    print(f"  Infinite values: {np.isinf(data.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Check data ranges
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Data Ranges:")
        for col in numeric_cols:
            print(f"  {col}: [{data[col].min():.3f}, {data[col].max():.3f}]")
    
    # Check categorical data
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\n📋 Categorical Data:")
        for col in categorical_cols:
            unique_vals = data[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 10:
                print(f"    Values: {list(data[col].unique())}")
    
    # Network structure validation
    print(f"\n🕸️ Network Validation:")
    print(f"  Nodes in network: {len(network.nodes())}")
    print(f"  Columns in data: {len(data.columns)}")
    print(f"  Nodes match columns: {set(network.nodes()) == set(data.columns)}")
    
    # Missing node-column mapping
    network_nodes = set(network.nodes())
    data_columns = set(data.columns)
    missing_in_data = network_nodes - data_columns
    extra_in_data = data_columns - network_nodes
    
    if missing_in_data:
        print(f"  ⚠️  Nodes missing in data: {missing_in_data}")
    if extra_in_data:
        print(f"  ⚠️  Extra columns in data: {extra_in_data}")

# Example usage
network, data = create_pgm(nodes=5, edges=8, samples=1000)
debug_data_quality(network, data)
```

#### 5. Performance Optimization

```python
# Issue: Slow performance with large datasets
import time
from functools import wraps

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Monitor key operations
@performance_monitor
def create_monitored_network(nodes, edges, samples):
    return create_pgm(nodes=nodes, edges=edges, samples=samples)

@performance_monitor  
def analyze_monitored_quality(network, data):
    quality = NetworkQualityMetrics(network, None, data)
    return quality.generate_comprehensive_report()

# Performance testing
print("🚀 PERFORMANCE TESTING")
print("=" * 25)

test_configs = [
    (5, 8, 1000),
    (10, 20, 2000), 
    (15, 30, 5000)
]

for nodes, edges, samples in test_configs:
    print(f"\nTesting: {nodes} nodes, {edges} edges, {samples} samples")
    network, data = create_monitored_network(nodes, edges, samples)
    report = analyze_monitored_quality(network, data)
```

---

## Best Practices

### 1. Project Organization

```bash
# Recommended project structure
my_bn_project/
├── data/
│   ├── raw/              # Original/benchmark data
│   ├── processed/        # Cleaned datasets  
│   └── synthetic/        # Generated datasets
├── networks/
│   ├── structures/       # Network definitions (.json, .bif)
│   ├── benchmarks/       # Standard benchmarks
│   └── custom/           # Custom networks
├── analysis/
│   ├── quality_reports/  # Quality assessment results
│   ├── visualizations/   # Plots and charts
│   └── comparisons/      # Comparative studies
├── scripts/
│   ├── generate_data.py  # Data generation scripts
│   ├── analyze_quality.py # Quality analysis scripts
│   └── create_networks.py # Network creation scripts
├── config/
│   ├── network_configs.json
│   └── analysis_params.json
├── requirements.txt
└── README.md
```

### 2. Configuration Management

```python
# config/network_configs.json
{
    "small_test": {
        "nodes": 5,
        "edges": 8,
        "samples": 1000,
        "missing_rate": 0.02,
        "noise_level": 0.01
    },
    "medium_research": {
        "nodes": 10,
        "edges": 20,
        "samples": 5000,
        "missing_rate": 0.05,
        "noise_level": 0.02
    },
    "large_production": {
        "nodes": 20,
        "edges": 50,
        "samples": 20000,
        "missing_rate": 0.03,
        "noise_level": 0.015
    }
}

# scripts/config_loader.py
import json
from bayesian_network_generator import create_comprehensive_pgm

def load_config(config_file, config_name):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs[config_name]

def create_from_config(config_file, config_name, seed=None):
    """Create network from configuration."""
    config = load_config(config_file, config_name)
    
    if seed:
        config['seed'] = seed
    
    return create_comprehensive_pgm(**config)

# Usage
results = create_from_config('config/network_configs.json', 'medium_research', seed=42)
```

### 3. Reproducible Research

```python
# scripts/reproducible_pipeline.py
import json
import pandas as pd
from datetime import datetime
from bayesian_network_generator import create_comprehensive_pgm, NetworkQualityMetrics

class ReproduciblePipeline:
    """Ensure reproducible research with full logging."""
    
    def __init__(self, experiment_name, seed=42):
        self.experiment_name = experiment_name
        self.seed = seed
        self.results_log = []
        self.start_time = datetime.now()
    
    def log_step(self, step_name, parameters, results):
        """Log each step of the pipeline."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'parameters': parameters,
            'results': results
        }
        self.results_log.append(log_entry)
    
    def create_network(self, **kwargs):
        """Create network with full logging."""
        kwargs['seed'] = self.seed
        
        print(f"🔧 Creating network with parameters: {kwargs}")
        results = create_comprehensive_pgm(**kwargs)
        
        network_info = {
            'nodes': len(results['network'].nodes()),
            'edges': len(results['network'].edges()),
            'data_shape': results['clean_data'].shape,
            'noisy_data_shape': results['noisy_data'].shape
        }
        
        self.log_step('create_network', kwargs, network_info)
        return results
    
    def analyze_quality(self, network, clean_data, noisy_data):
        """Analyze quality with logging."""
        print("🔍 Analyzing data quality...")
        
        quality = NetworkQualityMetrics(network, clean_data, noisy_data)
        report = quality.generate_comprehensive_report()
        
        quality_summary = {
            'overall_quality': report['overall_quality_score'],
            'completeness': report['completeness_score'],
            'consistency': report['consistency_score']
        }
        
        self.log_step('analyze_quality', {}, quality_summary)
        return report
    
    def save_results(self, results, output_dir='results'):
        """Save all results with metadata."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        results['clean_data'].to_csv(f'{output_dir}/{self.experiment_name}_clean.csv', index=False)
        results['noisy_data'].to_csv(f'{output_dir}/{self.experiment_name}_noisy.csv', index=False)
        
        # Save network structure
        import pickle
        with open(f'{output_dir}/{self.experiment_name}_network.pkl', 'wb') as f:
            pickle.dump(results['network'], f)
        
        # Save experiment log
        experiment_metadata = {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'steps': self.results_log
        }
        
        with open(f'{output_dir}/{self.experiment_name}_log.json', 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        print(f"✅ Results saved to {output_dir}/")
        return output_dir

# Example usage
pipeline = ReproduciblePipeline('medical_network_study', seed=42)

# Create network
results = pipeline.create_network(
    nodes=8,
    edges=15,
    samples=2000,
    missing_data_rate=0.05,
    noise_level=0.02
)

# Analyze quality
quality_report = pipeline.analyze_quality(
    results['network'],
    results['clean_data'], 
    results['noisy_data']
)

# Save everything
output_dir = pipeline.save_results(results)
print(f"📁 Experiment complete: {output_dir}")
```

### 4. Automated Testing & Validation

```python
# scripts/validation_suite.py
import pytest
from bayesian_network_generator import create_pgm, NetworkGenerator, NetworkQualityMetrics

class NetworkValidationSuite:
    """Comprehensive validation for generated networks."""
    
    @staticmethod
    def validate_network_structure(network, expected_nodes, expected_edges):
        """Validate basic network structure."""
        assert len(network.nodes()) == expected_nodes, f"Expected {expected_nodes} nodes, got {len(network.nodes())}"
        assert len(network.edges()) <= expected_edges, f"Too many edges: {len(network.edges())} > {expected_edges}"
        
        # Check for cycles (DAG requirement)
        import networkx as nx
        assert nx.is_directed_acyclic_graph(network), "Network contains cycles"
        
        print("✅ Network structure validation passed")
    
    @staticmethod
    def validate_data_quality(data, expected_samples, expected_columns):
        """Validate generated data quality."""
        assert data.shape[0] == expected_samples, f"Expected {expected_samples} samples, got {data.shape[0]}"
        assert data.shape[1] == expected_columns, f"Expected {expected_columns} columns, got {data.shape[1]}"
        
        # Check for basic data sanity
        assert not data.empty, "Data is empty"
        assert data.isnull().sum().sum() < data.size, "All data is null"
        
        print("✅ Data quality validation passed")
    
    @staticmethod
    def validate_reproducibility(seed, runs=3):
        """Validate reproducibility across multiple runs."""
        results = []
        
        for run in range(runs):
            network, data = create_pgm(nodes=5, edges=8, samples=100, seed=seed)
            results.append((network.edges(), data.values.tobytes()))
        
        # Check all runs produced identical results
        for i in range(1, runs):
            assert results[0][0] == results[i][0], f"Network structures differ between runs 0 and {i}"
            assert results[0][1] == results[i][1], f"Data differs between runs 0 and {i}"
        
        print(f"✅ Reproducibility validation passed ({runs} runs)")
    
    def run_full_validation(self):
        """Run complete validation suite."""
        print("🧪 RUNNING VALIDATION SUITE")
        print("=" * 35)
        
        # Test 1: Basic functionality
        print("\n🔬 Test 1: Basic Functionality")
        network, data = create_pgm(nodes=5, edges=8, samples=1000)
        self.validate_network_structure(network, 5, 8)
        self.validate_data_quality(data, 1000, 5)
        
        # Test 2: Large networks
        print("\n🔬 Test 2: Large Networks")
        network, data = create_pgm(nodes=15, edges=30, samples=2000)
        self.validate_network_structure(network, 15, 30)
        self.validate_data_quality(data, 2000, 15)
        
        # Test 3: Reproducibility
        print("\n🔬 Test 3: Reproducibility")
        self.validate_reproducibility(seed=42, runs=3)
        
        # Test 4: Quality metrics
        print("\n🔬 Test 4: Quality Metrics")
        generator = NetworkGenerator(seed=42)
        network = generator.create_random_network(nodes=6, edges=10)
        clean_data = generator.generate_data(network, samples=1000)
        noisy_data = generator.generate_data_with_quality_issues(
            network, samples=1000, missing_rate=0.05, noise_level=0.02
        )
        
        quality = NetworkQualityMetrics(network, clean_data, noisy_data)
        report = quality.generate_comprehensive_report()
        
        assert 'overall_quality_score' in report, "Missing overall quality score"
        assert 0 <= report['overall_quality_score'] <= 1, "Quality score out of range"
        
        print("✅ Quality metrics validation passed")
        
        print("\n🎉 ALL VALIDATIONS PASSED!")

# Run validation
validator = NetworkValidationSuite()
validator.run_full_validation()
```

### 5. Performance Benchmarking

```python
# scripts/benchmark_suite.py
import time
import psutil
import pandas as pd
from bayesian_network_generator import create_pgm, create_comprehensive_pgm

class PerformanceBenchmark:
    """Benchmark package performance across different configurations."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_creation(self, configurations):
        """Benchmark network creation performance."""
        print("⚡ PERFORMANCE BENCHMARKING")
        print("=" * 30)
        
        for config in configurations:
            print(f"\n🔧 Config: {config}")
            
            # Monitor system resources
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB
            start_time = time.time()
            
            try:
                # Create network
                if 'missing_data_rate' in config:
                    results = create_comprehensive_pgm(**config)
                    network = results['network']
                    data = results['noisy_data']
                else:
                    network, data = create_pgm(**config)
                
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024**2  # MB
                
                # Calculate metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                nodes_per_second = config['nodes'] / execution_time
                samples_per_second = config['samples'] / execution_time
                
                # Store results
                benchmark_result = {
                    'nodes': config['nodes'],
                    'edges': config['edges'],
                    'samples': config['samples'],
                    'execution_time': execution_time,
                    'memory_used_mb': memory_used,
                    'nodes_per_second': nodes_per_second,
                    'samples_per_second': samples_per_second,
                    'success': True
                }
                
                print(f"  ✅ Time: {execution_time:.2f}s, Memory: {memory_used:.1f}MB")
                print(f"     Rate: {nodes_per_second:.1f} nodes/s, {samples_per_second:.1f} samples/s")
                
            except Exception as e:
                benchmark_result = {
                    'nodes': config['nodes'],
                    'edges': config['edges'], 
                    'samples': config['samples'],
                    'execution_time': None,
                    'memory_used_mb': None,
                    'nodes_per_second': None,
                    'samples_per_second': None,
                    'success': False,
                    'error': str(e)
                }
                print(f"  ❌ Failed: {e}")
            
            self.results.append(benchmark_result)
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        df = pd.DataFrame(self.results)
        
        # Filter successful runs
        successful = df[df['success'] == True]
        
        if len(successful) == 0:
            print("❌ No successful benchmark runs!")
            return df
        
        print(f"\n📊 BENCHMARK SUMMARY ({len(successful)} successful runs)")
        print("=" * 50)
        
        # Performance statistics
        print(f"⏱️  Execution Time:")
        print(f"   Mean: {successful['execution_time'].mean():.2f}s")
        print(f"   Min:  {successful['execution_time'].min():.2f}s") 
        print(f"   Max:  {successful['execution_time'].max():.2f}s")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Mean: {successful['memory_used_mb'].mean():.1f}MB")
        print(f"   Min:  {successful['memory_used_mb'].min():.1f}MB")
        print(f"   Max:  {successful['memory_used_mb'].max():.1f}MB")
        
        print(f"\n🚀 Throughput:")
        print(f"   Avg nodes/second: {successful['nodes_per_second'].mean():.1f}")
        print(f"   Avg samples/second: {successful['samples_per_second'].mean():.1f}")
        
        # Save detailed results
        df.to_csv('benchmark_results.csv', index=False)
        print(f"\n📄 Detailed results saved to: benchmark_results.csv")
        
        return df

# Run benchmarks
benchmark = PerformanceBenchmark()

# Define test configurations
test_configs = [
    {'nodes': 5, 'edges': 8, 'samples': 1000},
    {'nodes': 10, 'edges': 20, 'samples': 2000}, 
    {'nodes': 15, 'edges': 30, 'samples': 5000},
    {'nodes': 20, 'edges': 40, 'samples': 10000},
    
    # With quality issues
    {'nodes': 10, 'edges': 15, 'samples': 2000, 'missing_data_rate': 0.05, 'noise_level': 0.02},
    {'nodes': 15, 'edges': 25, 'samples': 5000, 'missing_data_rate': 0.05, 'noise_level': 0.02}
]

benchmark.benchmark_creation(test_configs)
results_df = benchmark.generate_benchmark_report()
```

---

## 🎯 Quick Reference Summary

### Essential Commands
```bash
# Installation
pip install bayesian-network-generator

# Quick network creation
bng-create --nodes 5 --edges 8 --output network.json
bng-generate --network network.json --samples 1000 --output data.csv

# Quality analysis
bng-analyze --data data.csv --network network.json --output report.json
```

### Essential Functions
```python
from bayesian_network_generator import create_pgm, create_comprehensive_pgm, NetworkGenerator

# Basic usage
network, data = create_pgm(nodes=5, edges=8, samples=1000)

# Advanced usage
results = create_comprehensive_pgm(nodes=6, edges=10, samples=1500, 
                                 missing_data_rate=0.05, noise_level=0.02)

# Custom networks
generator = NetworkGenerator(seed=42)
custom_net = generator.create_chain_network(['A', 'B', 'C', 'D'])
```

### Common Workflows
1. **Research**: Create benchmark → Add quality issues → Analyze → Compare
2. **Testing**: Generate small networks → Validate structure → Test algorithms
3. **Production**: Load configuration → Create large networks → Quality control → Export

---

This comprehensive workflow guide provides everything needed to effectively use the Bayesian Network Generator package, from basic usage to advanced research scenarios. Each section includes practical examples that can be copied and adapted for specific use cases.

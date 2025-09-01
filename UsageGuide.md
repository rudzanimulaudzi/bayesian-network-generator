# 🚀 Complete Workflow Guide: Bayesian Network Generator

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [Installation & Setup](#installation--setup)
3. [Basic Workflow](#basic-workflow)
4. [Advanced Workflows](#advanced-workflows)
5. [CLI Command Reference](#cli-command-reference)
6. [Function API Reference](#function-api-reference)
7. [Real-World Examples](#real-world-examples)

---

## Quick Start Guide

### Setup & First Network

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

## Batch Network Generation with Parameter Loops

### Loop Setup: Basic Parameter Sweeps

```python
from bayesian_network_generator import create_pgm, create_comprehensive_pgm, NetworkGenerator
import pandas as pd
import numpy as np
import os
from itertools import product
import time

# Create output directory
os.makedirs('batch_networks', exist_ok=True)

print("🔄 BATCH NETWORK GENERATION")
print("=" * 35)

# Define parameter ranges
parameter_ranges = {
    'nodes': [5, 8, 10, 15, 20],
    'edges': [8, 12, 18, 25, 35],
    'samples': [1000, 2000, 5000],
    'missing_rates': [0.0, 0.02, 0.05, 0.10],
    'noise_levels': [0.0, 0.01, 0.03, 0.05]
}

# Simple parameter sweep
print("\n🔧 Simple Parameter Sweep:")
simple_configs = []

for nodes in parameter_ranges['nodes'][:3]:  # First 3 node sizes
    for samples in parameter_ranges['samples'][:2]:  # First 2 sample sizes
        config = {
            'nodes': nodes,
            'edges': int(nodes * 1.5),  # 1.5x edges per node
            'samples': samples,
            'seed': 42
        }
        simple_configs.append(config)

print(f"Generated {len(simple_configs)} simple configurations")

# Create networks from simple configs
simple_results = []
for i, config in enumerate(simple_configs):
    print(f"  Creating network {i+1}/{len(simple_configs)}: {config['nodes']} nodes, {config['samples']} samples")
    
    try:
        network, data = create_pgm(**config)
        
        result = {
            'config_id': i,
            'nodes': config['nodes'],
            'edges': len(network.edges()),
            'samples': config['samples'],
            'data_shape': data.shape,
            'success': True,
            'filename': f"batch_networks/simple_net_{i}.csv"
        }
        
        # Save data
        data.to_csv(result['filename'], index=False)
        simple_results.append(result)
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        simple_results.append({
            'config_id': i,
            'success': False,
            'error': str(e)
        })

print(f"✅ Simple sweep complete: {sum(1 for r in simple_results if r.get('success', False))} successful")
```

---

### Common Workflows
1. **Research**: Create benchmark → Add quality issues → Analyze → Compare
2. **Testing**: Generate small networks → Validate structure → Test algorithms
3. **Production**: Load configuration → Create large networks → Quality control → Export

This comprehensive workflow guide provides everything needed to effectively use the Bayesian Network Generator package, from basic usage to advanced research scenarios. Each section includes practical examples that can be copied and adapted for specific use cases.

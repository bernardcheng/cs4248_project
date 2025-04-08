from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import math

class Filter(ABC):
    @abstractmethod
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

class AverageWeightFilter(Filter):
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not edges:
            return []
        weights = np.array([edge['weight'] for edge in edges])
        avg_weight = np.mean(weights)
        return [edge for edge in edges if edge['weight'] >= avg_weight]

class WeightThresholdFilter(Filter):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if edge['weight'] >= self.threshold]

class MinMaxWeightFilter(Filter):
    def __init__(self, min_weight: float, max_weight: float):
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if self.min_weight <= edge['weight'] <= self.max_weight]

class RelationTypeFilter(Filter):
    def __init__(self, allowed_relations: List[str]):
        self.allowed_relations = allowed_relations
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if edge['rel']['@id'] in self.allowed_relations]

class DatasetFilter(Filter):
    def __init__(self, allowed_datasets: List[str]):
        self.allowed_datasets = allowed_datasets
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if edge['dataset'] in self.allowed_datasets]

class LanguageFilter(Filter):
    def __init__(self, allowed_languages: List[str]):
        self.allowed_languages = allowed_languages
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if any(lang in edge['end']['language'] for lang in self.allowed_languages)]

class EndNodeFilter(Filter):
    def __init__(self, predicate):
        self.predicate = predicate
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if self.predicate(edge['end'])]

class TopKFilter(Filter):
    def __init__(self, k: int):
        self.k = k
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(edges, key=lambda edge: edge['weight'], reverse=True)[:self.k]

class CompositeFilter(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = edges
        for f in self.filters:
            filtered = f(filtered)
        return filtered

class StdevFilter(Filter):
    def __init__(self, num_stdev: float = 1.0, keep_above: bool = True):
        self.num_stdev = num_stdev
        self.keep_above = keep_above
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not edges:
            return []
        weights = np.array([edge['weight'] for edge in edges])
        mean = np.mean(weights)
        stdev = np.std(weights, ddof=1) if len(weights) > 1 else 0
        threshold = mean + (self.num_stdev * stdev) if self.keep_above else mean - (self.num_stdev * stdev)
        
        if self.keep_above:
            return [edge for edge in edges if edge['weight'] >= threshold]
        return [edge for edge in edges if edge['weight'] <= threshold]

class FuzzyWeightFilter(Filter):
    def __init__(self, target_weight: float, tolerance: float = 0.2, smoothing: float = 0.1):
        self.target_weight = target_weight
        self.tolerance = tolerance
        self.smoothing = smoothing
    
    def _fuzzy_membership(self, weight: float) -> float:
        distance = abs(weight - self.target_weight)
        if distance > self.tolerance:
            return 0.0
        # Gaussian-like membership function
        return math.exp(-(distance ** 2) / (2 * self.smoothing ** 2))
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored_edges = [(edge, self._fuzzy_membership(edge['weight'])) for edge in edges]
        # Keep edges with non-zero membership scores
        return [edge for edge, score in scored_edges if score > 0.0]
    
class RemoveRelationTypeFilter(Filter):
    def __init__(self, disallowed_relations: List[str]):
        self.disallowed_relations = disallowed_relations
    
    def __call__(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [edge for edge in edges if edge['rel']['@id'] not in self.disallowed_relations]

# Pre-configured filter combinations
def create_default_filter() -> CompositeFilter:
    """Default filter to keep edges with above average weight"""
    return CompositeFilter([
        AverageWeightFilter(),
    ])

def create_average_weighted_relation_filter() -> CompositeFilter:
    """Keep edges with specific relation types and above average weight"""
    return CompositeFilter([
        AverageWeightFilter(),
        RelationTypeFilter([
            '/r/RelatedTo',
            '/r/Synonym', 
            '/r/IsA',
            '/r/HasProperty',
            '/r/CapableOf'
        ])
    ])

def create_basic_weight_filter(min_weight: float = 0.3) -> CompositeFilter:
    """Keep edges with weight above a threshold"""
    return CompositeFilter([
        WeightThresholdFilter(min_weight)
    ])

def create_high_quality_filter(top_k: int = 100) -> CompositeFilter:
    """Keep top-K weighted edges from specific datasets"""
    return CompositeFilter([
        AverageWeightFilter(),
        TopKFilter(top_k),
        DatasetFilter([
            '/d/wiktionary/en',
            '/d/wordnet/3.1'
        ])
    ])

def create_multilingual_weighted_filter(languages: List[str], min_weight: float, max_weight: float) -> CompositeFilter:
    """Keep edges that are multilingual and within a weight range"""
    return CompositeFilter([
        LanguageFilter(languages),
        MinMaxWeightFilter(min_weight, max_weight)
    ])

def create_relation_specific_filter(relations: List[str], min_weight: float = 0.5) -> CompositeFilter:
    """Keep edges that are of specific relation types and above a weight threshold"""
    return CompositeFilter([
        RelationTypeFilter(relations),
        WeightThresholdFilter(min_weight)
    ])

def create_dataset_specific_filter(datasets: List[str], top_k: int = 50) -> CompositeFilter:
    """Keep top-K weighted edges that are from specific datasets"""
    return CompositeFilter([
        DatasetFilter(datasets),
        AverageWeightFilter(),
        TopKFilter(top_k)
    ])

def create_comprehensive_filter(languages: List[str], relations: List[str], datasets: List[str], min_weight: float = 0.5, top_k: int = 50
) -> CompositeFilter:
    """Keep edges that pass multiple criteria - language, relation type, dataset, weight threshold, and top-K"""
    return CompositeFilter([
        LanguageFilter(languages),
        RelationTypeFilter(relations),
        DatasetFilter(datasets),
        WeightThresholdFilter(min_weight),
        TopKFilter(top_k)
    ])

def create_statistical_outlier_filter(num_stdev: float = 2.0, top_k: int = 50) -> CompositeFilter:
    """Keep top-K edges with weights more than num_stdev standard deviations above the mean"""
    return CompositeFilter([
        StdevFilter(num_stdev=num_stdev, keep_above=True),
        TopKFilter(top_k)
    ])

def create_semantic_similarity_filter(target_weight: float,languages: List[str] = ['en'],tolerance: float = 0.2) -> CompositeFilter:
    """Keep edges with weights similar to the target weight"""
    return CompositeFilter([
        LanguageFilter(languages),
        FuzzyWeightFilter(target_weight=target_weight, tolerance=tolerance)
    ])

def create_remove_relation_type_filter(disallowed_relations: List[str]) -> CompositeFilter:
    """Remove edges with specific relation types"""
    return CompositeFilter([
        RemoveRelationTypeFilter(disallowed_relations)
    ])
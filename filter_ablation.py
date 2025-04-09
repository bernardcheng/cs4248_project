from typing import List, Dict, Any
from scraper_filter import (
    CompositeFilter,
    create_default_filter,
    create_basic_weight_filter,
    create_average_weighted_relation_filter,
    create_high_quality_filter,
    create_multilingual_weighted_filter,
    create_relation_specific_filter,
    create_dataset_specific_filter,
    create_comprehensive_filter,
    create_statistical_outlier_filter,
    create_semantic_similarity_filter,
    create_remove_relation_type_filter
)

DEFAULT_MIN_WEIGHT = 0.3
DEFAULT_TOP_K = 50
DEFAULT_RELATIONS = ['/r/RelatedTo', '/r/Synonym', '/r/IsA', '/r/HasProperty', '/r/CapableOf']
DEFAULT_DATASETS = ['/d/wiktionary/en', '/d/wordnet/3.1']
DEFAULT_LANGUAGES = ['en']

filter_chains = {
    # weight threshold changes
    'baseline': create_default_filter(),
    'baseline_lenient': create_basic_weight_filter(min_weight=0.1),
    'baseline_strict': create_basic_weight_filter(min_weight=0.5),
    
    # statistical outlier variations
    'statistical_outliers': create_statistical_outlier_filter(num_stdev=2.0),
    'statistical_outliers_conservative': create_statistical_outlier_filter(num_stdev=1.0),
    'statistical_outliers_aggressive': create_statistical_outlier_filter(num_stdev=3.0),
    'statistical_outliers_top10': create_statistical_outlier_filter(num_stdev=2.0),
    'statistical_outliers_top100': create_statistical_outlier_filter(num_stdev=2.0),
    
    # relation-based filter variations
    'relation_specific': create_relation_specific_filter(
        relations=DEFAULT_RELATIONS
    ),
    'relation_minimal': create_relation_specific_filter(
        relations=['/r/IsA']
    ),
    'relation_minimal_synonyms': create_relation_specific_filter(
        relations=['/r/Synonym', '/r/IsA']
    ),
    'relation_minimal_synonyms_capabilities': create_relation_specific_filter(
        relations=['/r/Synonym', '/r/CapableOf', '/r/IsA']
    ),
    'relation_minimal_properties_synonyms_capabilities': create_relation_specific_filter(
        relations=['/r/HasProperty', '/r/Synonym', '/r/CapableOf', '/r/IsA']
    ),
    
    # dropping relation types
    'remove_relation_type_antonym': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym']
    ),
    'remove_relation_type_antonym_notdesires': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym', '/r/NotDesires']
    ),
    'remove_relation_type_antonym_notdesires_desires': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym', '/r/NotDesires', '/r/Desires']
    ),
    'remove_relation_type_antonym_notdesires_desires_obstructedby': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym', '/r/NotDesires', '/r/Desires', '/r/ObstructedBy']
    ),
    'remove_relation_type_antonym_notdesires_desires_obstructedby_mannerof': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym', '/r/NotDesires', '/r/Desires', '/r/ObstructedBy', '/r/MannerOf']
    ),
    'remove_relation_type_antonym_notdesires_desires_obstructedby_mannerof_causesdesire': create_remove_relation_type_filter(
        disallowed_relations=['/r/Antonym', '/r/NotDesires', '/r/Desires', '/r/ObstructedBy', '/r/MannerOf', '/r/CausesDesire']
    ),
    
    # dataset quality variations
    'high_quality': create_high_quality_filter(),
    
    # dataset source variations
    'dataset_specific': create_dataset_specific_filter(
        datasets=DEFAULT_DATASETS
    ),
    'dataset_wiktionary_only': create_dataset_specific_filter(
        datasets=['/d/wiktionary/en']
    ),
    'dataset_wordnet_only': create_dataset_specific_filter(
        datasets=['/d/wordnet/3.1']
    ),
    'dataset_wiktionary_wordnet': create_dataset_specific_filter(
        datasets=['/d/wiktionary/en', '/d/wordnet/3.1']
    ),
    
    # semantic similarity variations
    'semantic_similarity': create_semantic_similarity_filter(
        target_weight=0.5,
        tolerance=0.2
    ),
    'semantic_similarity_strict': create_semantic_similarity_filter(
        target_weight=0.5,
        tolerance=0.1
    ),
    'semantic_similarity_lenient': create_semantic_similarity_filter(
        target_weight=0.5,
        tolerance=0.3
    ),
    'semantic_similarity_high_weight': create_semantic_similarity_filter(
        target_weight=0.7,
        tolerance=0.2
    ),
    'semantic_similarity_low_weight': create_semantic_similarity_filter(
        target_weight=0.3,
        tolerance=0.2
    ),
    
    # comprehensive filter variations
    'comprehensive': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    'comprehensive_strict': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.5
    ),
    'comprehensive_lenient': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS + ['/r/DerivedFrom'],
        datasets=DEFAULT_DATASETS,
        min_weight=0.2
    ),
    'comprehensive_high_quality': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.5
    ),
    'comprehensive_high_quality_strict': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.7
    ),
}

def get_all_filter_chains() -> Dict[str, CompositeFilter]:
    return filter_chains

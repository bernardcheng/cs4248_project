from typing import List, Dict, Any
from scraper_filter import (
    CompositeFilter,
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
    'baseline': create_basic_weight_filter(min_weight=DEFAULT_MIN_WEIGHT),
    'baseline_strict': create_basic_weight_filter(min_weight=0.5),
    'baseline_lenient': create_basic_weight_filter(min_weight=0.1),
    'average_weight': create_average_weighted_relation_filter(),
    
    # statistical outlier variations
    'statistical_outliers': create_statistical_outlier_filter(num_stdev=2.0, top_k=DEFAULT_TOP_K),
    'statistical_outliers_conservative': create_statistical_outlier_filter(num_stdev=1.0, top_k=DEFAULT_TOP_K),
    'statistical_outliers_aggressive': create_statistical_outlier_filter(num_stdev=3.0, top_k=DEFAULT_TOP_K),
    'statistical_outliers_top10': create_statistical_outlier_filter(num_stdev=2.0, top_k=10),
    'statistical_outliers_top100': create_statistical_outlier_filter(num_stdev=2.0, top_k=100),
    
    # relation-based filter variations
    'relation_specific': create_relation_specific_filter(
        relations=DEFAULT_RELATIONS,
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    'relation_minimal': create_relation_specific_filter(
        relations=['/r/IsA'],
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    'relation_minimal_synonyms': create_relation_specific_filter(
        relations=['/r/Synonym', '/r/IsA'],
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    'relation_minimal_synonyms_capabilities': create_relation_specific_filter(
        relations=['/r/Synonym', '/r/CapableOf', '/r/IsA'],
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    'relation_minimal_properties_synonyms_capabilities': create_relation_specific_filter(
        relations=['/r/HasProperty', '/r/Synonym', '/r/CapableOf', '/r/IsA'],
        min_weight=DEFAULT_MIN_WEIGHT
    ),
    
    # dropping relation types
    'remove_relation_type_relatedto': create_remove_relation_type_filter(
        relations=['/r/RelatedTo'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'remove_relation_type_relatedto_synonym': create_remove_relation_type_filter(
        relations=['/r/RelatedTo', '/r/Synonym'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'remove_relation_type_relatedto_synonym_isa': create_remove_relation_type_filter(
        relations=['/r/RelatedTo', '/r/Synonym', '/r/IsA'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'remove_relation_type_relatedto_synonym_isa_hasproperty': create_remove_relation_type_filter(
        relations=['/r/RelatedTo', '/r/Synonym', '/r/IsA', '/r/HasProperty'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'remove_relation_type_relatedto_synonym_isa_hasproperty_capableof': create_remove_relation_type_filter(
        relations=['/r/RelatedTo', '/r/Synonym', '/r/IsA', '/r/HasProperty', '/r/CapableOf'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'remove_relation_type_relatedto_synonym_isa_hasproperty_capableof_derivedfrom': create_remove_relation_type_filter(
        relations=['/r/RelatedTo', '/r/Synonym', '/r/IsA', '/r/HasProperty', '/r/CapableOf', '/r/DerivedFrom'],
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    
    # dataset quality variations
    'high_quality': create_high_quality_filter(top_k=DEFAULT_TOP_K),
    'high_quality_selective': create_high_quality_filter(top_k=25),
    'high_quality_broad': create_high_quality_filter(top_k=100),
    
    # dataset source variations
    'dataset_specific': create_dataset_specific_filter(
        datasets=DEFAULT_DATASETS,
        top_k=DEFAULT_TOP_K
    ),
    'dataset_wiktionary_only': create_dataset_specific_filter(
        datasets=['/d/wiktionary/en'],
        top_k=DEFAULT_TOP_K
    ),
    'dataset_wordnet_only': create_dataset_specific_filter(
        datasets=['/d/wordnet/3.1'],
        top_k=DEFAULT_TOP_K
    ),
    'dataset_wiktionary_wordnet': create_dataset_specific_filter(
        datasets=['/d/wiktionary/en', '/d/wordnet/3.1'],
        top_k=DEFAULT_TOP_K
    ),
    
    # semantic similarity variations
    'semantic_similarity': create_semantic_similarity_filter(
        target_weight=0.5,
        languages=DEFAULT_LANGUAGES,
        tolerance=0.2
    ),
    'semantic_similarity_strict': create_semantic_similarity_filter(
        target_weight=0.5,
        languages=DEFAULT_LANGUAGES,
        tolerance=0.1
    ),
    'semantic_similarity_lenient': create_semantic_similarity_filter(
        target_weight=0.5,
        languages=DEFAULT_LANGUAGES,
        tolerance=0.3
    ),
    'semantic_similarity_high_weight': create_semantic_similarity_filter(
        target_weight=0.7,
        languages=DEFAULT_LANGUAGES,
        tolerance=0.2
    ),
    'semantic_similarity_low_weight': create_semantic_similarity_filter(
        target_weight=0.3,
        languages=DEFAULT_LANGUAGES,
        tolerance=0.2
    ),
    
    # comprehensive filter variations
    'comprehensive': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=DEFAULT_MIN_WEIGHT,
        top_k=DEFAULT_TOP_K
    ),
    'comprehensive_strict': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.5,
        top_k=25
    ),
    'comprehensive_lenient': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS + ['/r/DerivedFrom'],
        datasets=DEFAULT_DATASETS,
        min_weight=0.2,
        top_k=100
    ),
    'comprehensive_high_quality': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.5,
        top_k=50
    ),
    'comprehensive_high_quality_strict': create_comprehensive_filter(
        languages=DEFAULT_LANGUAGES,
        relations=DEFAULT_RELATIONS,
        datasets=DEFAULT_DATASETS,
        min_weight=0.7,
        top_k=25
    ),
}

def get_all_filter_chains() -> Dict[str, CompositeFilter]:
    return filter_chains

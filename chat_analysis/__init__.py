# Chat Analysis - Queued Batch Processing
"""
Conversation analysis using queued batch processing for optimal performance.

Main entry point: analyze_conversations_with_queued_batching()
"""

# Main analysis function - the primary entry point
from .template_analysis_enhanced import analyze_conversations_with_queued_batching

# EDA analysis utilities  
from .eda import run_eda_analysis

# Template utilities
from .bio_template import create_empty_template, template_to_human_readable

# For advanced users who need direct access to components
from .queue_system import api_queue
from .extraction_strategies import (
    queue_individual_extraction_request,
    queue_sectioned_aggregation_requests,
    process_individual_extraction_results,
    process_aggregation_results
)

__all__ = [
    'analyze_conversations_with_queued_batching',
    'run_eda_analysis', 
    'create_empty_template',
    'template_to_human_readable',
    'api_queue',
    'queue_individual_extraction_request',
    'queue_sectioned_aggregation_requests', 
    'process_individual_extraction_results',
    'process_aggregation_results'
]

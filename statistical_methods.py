"""Statistical methods used in the vessel-imaging manuscript.

These functions expose the models separately from the optional web interface.
Each function accepts the observations for the comparison being assessed; the
statistical unit, variable names and return values are described below.
No study observations or fitted results are embedded in this module.
"""
from core.paper_helpers import (
    session_values,
    mouse_compartment_values,
    fit_pacap_compartment,
    fit_baseline_compartment,
    fit_paper_compartment_slope,
)
from core.diameter_helpers import build_vessel_values, linear_relationship_stats
from core.mixed import (
    mixed_model_size_interaction,
    mixed_model_two_group_mouse_intercept,
    mixed_model_two_group_category_interaction,
)
from core.stats import holm_adjust


def treatment_by_compartment(mouse_session_values, reference, comparison):
    """Treatment × compartment model and two Holm-adjusted planned contrasts.

    Input: one mean per mouse/session, treatment and compartment. Required
    columns: z_response, group, mouse_alias, dataset_id, compartment (dura/pia).
    The model accounts for repeated measurements using a mouse random intercept.
    Returns model input, Wald interaction table, planned contrasts, model
    diagnostics and method description, in that order.
    """
    return mixed_model_two_group_category_interaction(
        mouse_session_values,
        reference=reference,
        comparison=comparison,
        category_col='compartment',
        categories=['dura','pia'],
        category_label='Vessel compartment',
    )


__all__ = [
    'session_values', 'mouse_compartment_values', 'fit_pacap_compartment',
    'fit_baseline_compartment', 'fit_paper_compartment_slope',
    'build_vessel_values', 'linear_relationship_stats',
    'mixed_model_size_interaction', 'mixed_model_two_group_mouse_intercept',
    'mixed_model_two_group_category_interaction', 'treatment_by_compartment',
    'holm_adjust',
]

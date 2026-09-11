from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.fresh_start import build_fresh_head
from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
    MechanisticStrongBaselineProcessHead,
)
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.output_applicability import (
    GSMEAN_OUTPUT_FIELD,
    apply_gsmean_output_applicability,
    build_gsmean_output_applicability,
)
from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
    MechanisticTargetOrientedAttentionProcessHead,
)
from research.daily_coarse_graining.daily_operator.types import PFTAxisInput


def _schema(
    n_pft: int,
    *,
    full_pft_axis: bool = False,
    active_pft_indices: tuple[int, ...] | None = None,
) -> NeuralOutputSchema:
    if active_pft_indices is None:
        active_pft_indices = tuple(range(n_pft))
    return NeuralOutputSchema(
        fields=(
            OutputFieldSpec(
                name=GSMEAN_OUTPUT_FIELD,
                shape=(1, n_pft),
                label_index=0,
                constraint="nonnegative",
                pft_axis=1,
                full_pft_axis=full_pft_axis,
            ),
        ),
        labels=("fast_owner.diffuco",),
        active_pft_indices=active_pft_indices,
    )


def _inputs(traits: np.ndarray, active: np.ndarray | None = None):
    n_pft = traits.shape[0]
    pft = PFTAxisInput(
        pft_state=np.zeros((n_pft, 1), dtype=np.float64),
        pft_parameters=np.zeros((n_pft, 1), dtype=np.float64),
        pft_traits=traits,
        pft_fraction=np.ones((n_pft,), dtype=np.float64),
        active_pft_mask=(
            np.ones((n_pft,), dtype=np.bool_) if active is None else active
        ),
    )
    return SimpleNamespace(pft=pft)


def _named_trait_routing(index: int):
    channels = tuple(
        SimpleNamespace(index=position, name=name)
        for position, name in enumerate(
            ("is_tree", "natural", "role_bare_soil")
        )
    )
    assert channels[index].name == "role_bare_soil"
    return build_gsmean_output_applicability(
        SimpleNamespace(trait_channels=channels)
    )


def test_named_bare_soil_trait_zeroes_only_the_matching_pft():
    applicability = _named_trait_routing(2)
    traits = np.zeros((3, 3), dtype=np.float64)
    traits[1, 2] = 1.0
    outputs = {
        GSMEAN_OUTPUT_FIELD: jnp.asarray([[2.0, 2769.0, 4.0]]),
        "unchanged": jnp.asarray([7.0]),
    }

    corrected = apply_gsmean_output_applicability(
        outputs,
        _inputs(traits),
        schema=_schema(3),
        applicability=applicability,
    )

    np.testing.assert_array_equal(
        np.asarray(corrected[GSMEAN_OUTPUT_FIELD]),
        np.asarray([[2.0, 0.0, 4.0]]),
    )
    assert corrected["unchanged"] is outputs["unchanged"]


def test_bare_soil_value_and_parameter_jacobian_are_exactly_zero():
    applicability = _named_trait_routing(2)
    traits = np.zeros((3, 3), dtype=np.float64)
    traits[1, 2] = 1.0
    inputs = _inputs(traits)
    schema = _schema(3)

    def corrected_values(parameters):
        outputs = {GSMEAN_OUTPUT_FIELD: (parameters**2)[None, :]}
        return apply_gsmean_output_applicability(
            outputs,
            inputs,
            schema=schema,
            applicability=applicability,
        )[GSMEAN_OUTPUT_FIELD][0]

    parameters = jnp.asarray([2.0, 3.0, 4.0])
    values = corrected_values(parameters)
    jacobian = jax.jacrev(corrected_values)(parameters)

    np.testing.assert_array_equal(np.asarray(values), np.asarray([4.0, 0.0, 16.0]))
    np.testing.assert_array_equal(np.asarray(jacobian[1]), np.zeros((3,)))
    np.testing.assert_array_equal(
        np.asarray(jacobian)[[0, 2]],
        np.asarray([[4.0, 0.0, 0.0], [0.0, 0.0, 8.0]]),
    )


def test_applicability_is_permutation_equivariant_and_does_not_mask_inactive_vegetation():
    applicability = _named_trait_routing(2)
    traits = np.zeros((3, 3), dtype=np.float64)
    traits[1, 2] = 1.0
    active = np.asarray([True, True, False])
    values = np.asarray([[2.0, 2769.0, 99.0]])
    original = apply_gsmean_output_applicability(
        {GSMEAN_OUTPUT_FIELD: values},
        _inputs(traits, active),
        schema=_schema(3),
        applicability=applicability,
    )[GSMEAN_OUTPUT_FIELD]

    order = np.asarray([2, 0, 1])
    permuted = apply_gsmean_output_applicability(
        {GSMEAN_OUTPUT_FIELD: values[:, order]},
        _inputs(traits[order], active[order]),
        schema=_schema(3),
        applicability=applicability,
    )[GSMEAN_OUTPUT_FIELD]

    np.testing.assert_array_equal(np.asarray(permuted), np.asarray(original)[:, order])
    assert float(original[0, 2]) == 99.0


def test_full_pft_layout_uses_the_declared_compact_to_full_mapping():
    applicability = _named_trait_routing(2)
    traits = np.zeros((3, 3), dtype=np.float64)
    traits[1, 2] = 1.0
    values = np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0]])

    corrected = apply_gsmean_output_applicability(
        {GSMEAN_OUTPUT_FIELD: values},
        _inputs(traits),
        schema=_schema(
            5,
            full_pft_axis=True,
            active_pft_indices=(4, 1, 3),
        ),
        applicability=applicability,
    )[GSMEAN_OUTPUT_FIELD]

    np.testing.assert_array_equal(
        np.asarray(corrected),
        np.asarray([[10.0, 0.0, 30.0, 40.0, 50.0]]),
    )


def test_enabled_target_head_passes_corrected_gsmean_to_reconstruction_boundary(
    monkeypatch,
):
    traits = np.zeros((3, 1), dtype=np.float64)
    traits[1, 0] = 1.0
    inputs = _inputs(traits)
    outputs = {GSMEAN_OUTPUT_FIELD: jnp.asarray([[2.0, 2769.0, 4.0]])}
    captured = {}
    sentinel = object()

    def capture_reconstruction(_head, learned_outputs, passed_inputs):
        captured.update(learned_outputs)
        assert passed_inputs is inputs
        return sentinel

    monkeypatch.setattr(
        MechanisticStrongBaselineProcessHead,
        "prediction_from_learned_outputs",
        capture_reconstruction,
    )
    head = object.__new__(MechanisticTargetOrientedAttentionProcessHead)
    object.__setattr__(head, "apply_gsmean_applicability", True)
    object.__setattr__(head, "learned_schema", _schema(3))

    result = head.prediction_from_learned_outputs(outputs, inputs)

    assert result is sentinel
    np.testing.assert_array_equal(
        np.asarray(captured[GSMEAN_OUTPUT_FIELD]),
        np.asarray([[2.0, 0.0, 4.0]]),
    )


def test_gsmean_correction_is_explicitly_default_off():
    assert (
        MechanisticTargetOrientedAttentionProcessHead.__dataclass_fields__[
            "apply_gsmean_applicability"
        ].default
        is False
    )
    parameter = signature(build_fresh_head).parameters["apply_gsmean_applicability"]
    assert parameter.default is False

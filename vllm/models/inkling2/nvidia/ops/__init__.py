# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

from vllm.triton_utils import tl


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def use_pdl() -> bool:
    """Whether kernels use programmatic dependent launch (Hopper and newer).

    `griddepcontrol` does not assemble below sm_90, so older cards launch
    without it and the in-kernel wait and trigger become no-ops."""
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


@tl.core.builtin
def gdc_wait(_semantic=None):
    if use_pdl():
        return tl.extra.cuda.gdc_wait(_semantic=_semantic)


@tl.core.builtin
def gdc_launch_dependents(_semantic=None):
    if use_pdl():
        return tl.extra.cuda.gdc_launch_dependents(_semantic=_semantic)

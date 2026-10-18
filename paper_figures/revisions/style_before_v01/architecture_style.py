"""Shared color roles for the diffusion and residual architecture figures."""

ARCHITECTURE_COLORS = {
    'learned': '#72B6F2',       # Convolutions, learned enlargement, network blocks.
    'nisp_condition': '#55CFB0',  # NISP images and conditioning/modulation.
    'jwst_image': '#F3BE57',    # Observed JWST targets and F115W estimates.
    'diffusion': '#B39AE8',     # Timestep, velocity representation, DDIM update.
    'neutral': '#E2E8F0',       # Noise, losses, fixed resize, generic feature states.
    'ink': '#192C3B',
}

ARCHITECTURE_BORDERS = {
    '#72B6F2': '#2872B2',
    '#55CFB0': '#087B64',
    '#F3BE57': '#A97416',
    '#B39AE8': '#7755AF',
    '#E2E8F0': '#718198',
}

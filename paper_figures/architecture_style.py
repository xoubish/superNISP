"""Shared semantic colors and readable text for the architecture figures."""
from matplotlib.colors import to_rgb, to_hex
from figure_utils import COLORS, INK

ARCHITECTURE_COLORS = {
    'learned': COLORS['legacy'],
    'nisp_condition': COLORS['native'],
    'jwst_image': COLORS['jwst'],
    'diffusion': COLORS['diffusion'],
    'direct5': COLORS['direct5'],
    'neutral': '#F0F0F0',
    'ink': INK,
}
ARCHITECTURE_BORDERS = {color:to_hex([v*.7 for v in to_rgb(color)])
                        for role,color in ARCHITECTURE_COLORS.items() if role!='ink'}

def contrast_text(color):
    rgb=to_rgb(color)
    linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    luminance=sum(v*w for v,w in zip(linear,[.2126,.7152,.0722]))
    return 'white' if 1.05/(luminance+.05)>=(luminance+.05)/.05 else '#111111'

ARCHITECTURE_TEXT={color:contrast_text(color) for color in ARCHITECTURE_BORDERS}

from django import template

register = template.Library()


@register.filter
def percent(value, digits: int = 1) -> str:
    """0.123 -> "12.3 %"."""
    if value is None:
        return ""
    return f"{float(value) * 100:.{int(digits)}f} %"


@register.filter
def get(mapping, key):
    """Dictionary lookup with a variable key: ``{{ counts|get:kind }}``."""
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter
def thousands(value) -> str:
    """12345 -> "12,345"."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return ""

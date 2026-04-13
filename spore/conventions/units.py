import pint

# Shared unit registry — import this instead of creating a new UnitRegistry
# elsewhere, so that Quantity comparisons across modules work correctly.
ureg = pint.UnitRegistry()

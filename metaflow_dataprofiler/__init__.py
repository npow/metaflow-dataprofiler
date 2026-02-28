"""
metaflow_dataprofiler
=====================
Public API for the metaflow-dataprofiler extension.

The primary export is :class:`DataProfileComponent`, which lets you embed
a ydata-profiling report inside any Metaflow card via ``current.card.append()``.

Example::

    from metaflow_dataprofiler import DataProfileComponent

    @card
    @step
    def clean(self):
        raw = load_data()
        current.card.append(DataProfileComponent(raw, title="Before cleaning"))
        self.df = clean(raw)
        current.card.append(DataProfileComponent(self.df, title="After cleaning"))
"""

from metaflow_extensions.dataprofile.plugins.cards.dataprofile import (
    DataProfileComponent,
)

__all__ = ["DataProfileComponent"]

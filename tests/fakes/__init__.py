"""Shared test fakes for jamma.

A fake is a lightweight class implementing the real interface of a
collaborator. Unlike ``MagicMock``, accessing an undeclared attribute
raises ``AttributeError`` — catching interface drift the moment a method
is renamed or removed.

See docs/TESTING.md §2.3 for the policy.
"""

from tests.fakes.assoc_writer import FakeAssocWriter
from tests.fakes.progress import FakeProgressBar, FakeProgressbarModule

__all__ = ["FakeAssocWriter", "FakeProgressBar", "FakeProgressbarModule"]

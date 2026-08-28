"""Shared test fakes for jamma.

A fake is a lightweight class implementing the real interface of a
collaborator. Unlike ``MagicMock``, accessing an undeclared attribute
raises ``AttributeError`` — catching interface drift the moment a method
is renamed or removed.

See docs/TESTING.md §2.3 for the policy.
"""

from tests.fakes.assoc_writer import FakeAssocWriter
from tests.fakes.jlinalg import FakeJlinalg, use_fake_jlinalg
from tests.fakes.pipeline import FakePipelineRunner, FakePipelineRunnerFactory
from tests.fakes.progress import FakeProgressBar, FakeProgressbarModule

__all__ = [
    "FakeAssocWriter",
    "FakeJlinalg",
    "FakePipelineRunner",
    "FakePipelineRunnerFactory",
    "FakeProgressBar",
    "FakeProgressbarModule",
    "use_fake_jlinalg",
]

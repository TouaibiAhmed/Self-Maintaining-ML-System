"""Let monitoring runs record insufficient-data and failed outcomes honestly.

drift_share, drift_detected and the window bounds become nullable so a run that could not
measure drift stores NULL ("unknown") instead of a misleading 0 / false. finished_at and error
record when a run ended and why it failed.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_monitoring_run_outcomes"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NULLABLE_COLUMNS = {
    "window_start": sa.DateTime(timezone=True),
    "window_end": sa.DateTime(timezone=True),
    "drift_share": sa.Float(),
    "drift_detected": sa.Boolean(),
}


def upgrade() -> None:
    for name, column_type in NULLABLE_COLUMNS.items():
        op.alter_column("monitoring_runs", name, existing_type=column_type, nullable=True)
    op.add_column("monitoring_runs", sa.Column("finished_at", sa.DateTime(timezone=True)))
    op.add_column("monitoring_runs", sa.Column("error", sa.Text()))


def downgrade() -> None:
    op.drop_column("monitoring_runs", "error")
    op.drop_column("monitoring_runs", "finished_at")
    op.execute("DELETE FROM monitoring_runs WHERE drift_share IS NULL OR window_start IS NULL")
    for name, column_type in NULLABLE_COLUMNS.items():
        op.alter_column("monitoring_runs", name, existing_type=column_type, nullable=False)

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0001_init_web_stack"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "teams",
        sa.Column("team_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("logo_url", sa.Text(), nullable=True),
        sa.Column("is_tracked", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_teams_is_tracked", "teams", ["is_tracked"], unique=False)

    op.create_table(
        "team_aliases",
        sa.Column("alias_team_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column(
            "canonical_team_id",
            sa.BigInteger(),
            sa.ForeignKey("teams.team_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_team_aliases_canonical_team_id", "team_aliases", ["canonical_team_id"], unique=False)

    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("run_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("valid", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("stats", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_runs_run_type", "runs", ["run_type"], unique=False)
    op.create_index("ix_runs_status", "runs", ["status"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("rq_job_id", sa.Text(), nullable=True),
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("logs", sa.Text(), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_rq_job_id", "jobs", ["rq_job_id"], unique=False)

    op.create_table(
        "matches",
        sa.Column("match_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("start_time", sa.BigInteger(), nullable=True),
        sa.Column("leagueid", sa.BigInteger(), nullable=True),
        sa.Column("series_id", sa.BigInteger(), nullable=True),
        sa.Column("radiant_team_id", sa.BigInteger(), nullable=True),
        sa.Column("dire_team_id", sa.BigInteger(), nullable=True),
        sa.Column("radiant_win", sa.Boolean(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_matches_start_time", "matches", ["start_time"], unique=False)
    op.create_index("ix_matches_leagueid", "matches", ["leagueid"], unique=False)
    op.create_index("ix_matches_series_id", "matches", ["series_id"], unique=False)
    op.create_index("ix_matches_radiant_team_id", "matches", ["radiant_team_id"], unique=False)
    op.create_index("ix_matches_dire_team_id", "matches", ["dire_team_id"], unique=False)

    op.create_table(
        "series",
        sa.Column("leagueid", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("series_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("start_time_min", sa.BigInteger(), nullable=True),
        sa.Column("start_time_max", sa.BigInteger(), nullable=True),
        sa.Column("team_a", sa.BigInteger(), nullable=True),
        sa.Column("team_b", sa.BigInteger(), nullable=True),
        sa.Column("score_team_a", sa.Integer(), nullable=True),
        sa.Column("score_team_b", sa.Integer(), nullable=True),
        sa.Column("winner_team_id", sa.BigInteger(), nullable=True),
        sa.Column("bo_type", sa.Integer(), nullable=True),
        sa.Column("series_type_raw", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.UniqueConstraint("leagueid", "series_id", name="uq_series_league_series"),
    )
    op.create_index("ix_series_start_time_min", "series", ["start_time_min"], unique=False)
    op.create_index("ix_series_team_a", "series", ["team_a"], unique=False)
    op.create_index("ix_series_team_b", "series", ["team_b"], unique=False)
    op.create_index("ix_series_winner_team_id", "series", ["winner_team_id"], unique=False)

    op.create_table(
        "extras",
        sa.Column("match_id", sa.BigInteger(), primary_key=True, nullable=False),
        sa.Column("radiant_gold_adv", postgresql.JSONB(), nullable=True),
        sa.Column("radiant_xp_adv", postgresql.JSONB(), nullable=True),
        sa.Column("picks_bans", postgresql.JSONB(), nullable=True),
    )

    op.create_table(
        "players",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=True),
        sa.Column("player_slot", sa.Integer(), nullable=True),
        sa.Column("account_id", sa.BigInteger(), nullable=True),
        sa.Column("hero_id", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_players_match_id", "players", ["match_id"], unique=False)
    op.create_index("ix_players_player_slot", "players", ["player_slot"], unique=False)
    op.create_index("ix_players_account_id", "players", ["account_id"], unique=False)
    op.create_index("ix_players_hero_id", "players", ["hero_id"], unique=False)

    op.create_table(
        "objectives",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=True),
        sa.Column("time", sa.Integer(), nullable=True),
        sa.Column("objective_type", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_objectives_match_id", "objectives", ["match_id"], unique=False)
    op.create_index("ix_objectives_time", "objectives", ["time"], unique=False)
    op.create_index("ix_objectives_objective_type", "objectives", ["objective_type"], unique=False)

    op.create_table(
        "teamfights",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=True),
        sa.Column("teamfight_index", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_teamfights_match_id", "teamfights", ["match_id"], unique=False)
    op.create_index("ix_teamfights_teamfight_index", "teamfights", ["teamfight_index"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_teamfights_teamfight_index", table_name="teamfights")
    op.drop_index("ix_teamfights_match_id", table_name="teamfights")
    op.drop_table("teamfights")

    op.drop_index("ix_objectives_objective_type", table_name="objectives")
    op.drop_index("ix_objectives_time", table_name="objectives")
    op.drop_index("ix_objectives_match_id", table_name="objectives")
    op.drop_table("objectives")

    op.drop_index("ix_players_hero_id", table_name="players")
    op.drop_index("ix_players_account_id", table_name="players")
    op.drop_index("ix_players_player_slot", table_name="players")
    op.drop_index("ix_players_match_id", table_name="players")
    op.drop_table("players")

    op.drop_table("extras")

    op.drop_index("ix_series_winner_team_id", table_name="series")
    op.drop_index("ix_series_team_b", table_name="series")
    op.drop_index("ix_series_team_a", table_name="series")
    op.drop_index("ix_series_start_time_min", table_name="series")
    op.drop_table("series")

    op.drop_index("ix_matches_dire_team_id", table_name="matches")
    op.drop_index("ix_matches_radiant_team_id", table_name="matches")
    op.drop_index("ix_matches_series_id", table_name="matches")
    op.drop_index("ix_matches_leagueid", table_name="matches")
    op.drop_index("ix_matches_start_time", table_name="matches")
    op.drop_table("matches")

    op.drop_index("ix_jobs_rq_job_id", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_job_type", table_name="jobs")
    op.drop_table("jobs")

    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_run_type", table_name="runs")
    op.drop_table("runs")

    op.drop_index("ix_team_aliases_canonical_team_id", table_name="team_aliases")
    op.drop_table("team_aliases")

    op.drop_index("ix_teams_is_tracked", table_name="teams")
    op.drop_table("teams")

    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")


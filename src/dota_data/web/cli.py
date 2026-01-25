from __future__ import annotations

import argparse
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from .auth import get_password_hash
from .bootstrap import BootstrapPaths, bootstrap_processed_tables, upsert_teams_and_aliases
from .db import SessionLocal, engine
from .models import User


def _alembic_cfg() -> Config:
    cfg = Config("alembic.ini")
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def cmd_migrate(_: argparse.Namespace) -> None:
    command.upgrade(_alembic_cfg(), "head")


def cmd_db_reset(_: argparse.Namespace) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_cfg(), "head")


def cmd_create_user(args: argparse.Namespace) -> None:
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == args.username).first() is not None:
            raise SystemExit(f"User {args.username!r} already exists")
        u = User(username=args.username, password_hash=get_password_hash(args.password), is_admin=bool(args.admin))
        db.add(u)
        db.commit()
        print(f"Created user: {u.username} (admin={u.is_admin})")
    finally:
        db.close()


def cmd_bootstrap(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed).resolve()
    teams_csv = Path(args.teams_csv).resolve()
    aliases_csv = Path(args.aliases_csv).resolve()

    if args.reset:
        cmd_db_reset(args)

    db = SessionLocal()
    try:
        upsert_teams_and_aliases(db, BootstrapPaths(processed_dir=processed_dir, teams_csv=teams_csv, aliases_csv=aliases_csv))
        counts = bootstrap_processed_tables(
            db,
            processed_dir,
            batch_size=int(args.batch_size),
            include_big=not bool(args.skip_big),
        )
        db.commit()
        print("Bootstrap done:", counts)
    finally:
        db.close()


def cmd_import_teams(args: argparse.Namespace) -> None:
    teams_csv = Path(args.teams_csv).resolve()
    aliases_csv = Path(args.aliases_csv).resolve()
    db = SessionLocal()
    try:
        upsert_teams_and_aliases(
            db, BootstrapPaths(processed_dir=Path("."), teams_csv=teams_csv, aliases_csv=aliases_csv)
        )
        db.commit()
        print("Imported teams + aliases.")
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dota-data-web")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("migrate", help="Run Alembic migrations (upgrade head)")
    sp.set_defaults(func=cmd_migrate)

    sp = sub.add_parser("db-reset", help="Drop and recreate public schema, then migrate")
    sp.set_defaults(func=cmd_db_reset)

    sp = sub.add_parser("create-user", help="Create a user (admin-only workflow for now)")
    sp.add_argument("--username", required=True)
    sp.add_argument("--password", required=True)
    sp.add_argument("--admin", action="store_true")
    sp.set_defaults(func=cmd_create_user)

    sp = sub.add_parser("bootstrap", help="Bootstrap DB from existing processed parquet + CSVs")
    sp.add_argument("--processed", default="data/processed")
    sp.add_argument("--teams-csv", default="data/teams_to_look.csv")
    sp.add_argument("--aliases-csv", default="data/team_aliases.csv")
    sp.add_argument("--batch-size", default="10000")
    sp.add_argument("--reset", action="store_true", help="Reset DB schema before bootstrapping")
    sp.add_argument("--skip-big", action="store_true", help="Skip players/objectives/teamfights (fast smoke test)")
    sp.set_defaults(func=cmd_bootstrap)

    sp = sub.add_parser("import-teams", help="Import teams + aliases from CSVs (no processed parquet)")
    sp.add_argument("--teams-csv", default="data/teams_to_look.csv")
    sp.add_argument("--aliases-csv", default="data/team_aliases.csv")
    sp.set_defaults(func=cmd_import_teams)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

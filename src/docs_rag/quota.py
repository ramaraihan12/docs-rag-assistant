"""Question limits for the public demo, counted in Postgres.

Serverless instances share no memory, so a counter in the process would start over on
every cold start and count each instance separately. Each hit here is one atomic upsert,
which keeps the numbers right when several instances touch the same row at once.
"""

import ipaddress
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sqlalchemy import BigInteger, Column, Engine, Integer, MetaData, String, Table, text

MINUTE = 60
DAY = 86_400


@dataclass(frozen=True)
class Refusal:
    message: str
    retry_after: int


class Quota:
    def __init__(
        self,
        engine: Engine,
        table_name: str,
        *,
        per_minute: int,
        per_day: int,
        all_per_day: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Every statement commits by itself, so a request never holds two counters locked
        # at once and two requests can't deadlock on them.
        self._engine = engine.execution_options(isolation_level="AUTOCOMMIT")
        self._per_minute = per_minute
        self._per_day = per_day
        self._all_per_day = all_per_day
        self._clock = clock

        metadata = MetaData()
        table = Table(
            table_name,
            metadata,
            Column("key", String(64), primary_key=True),
            Column("hits", Integer, nullable=False),
            Column("expires_at", BigInteger, nullable=False),
        )
        metadata.create_all(engine)

        name = engine.dialect.identifier_preparer.format_table(table)
        self._hit = text(f"""
            INSERT INTO {name} (key, hits, expires_at) VALUES (:key, 1, :expires_at)
            ON CONFLICT (key) DO UPDATE SET
                hits = CASE WHEN {name}.expires_at <= :now THEN 1 ELSE {name}.hits + 1 END,
                expires_at = CASE WHEN {name}.expires_at <= :now
                    THEN excluded.expires_at ELSE {name}.expires_at END
            RETURNING hits, expires_at
        """)
        self._sweep = text(f"DELETE FROM {name} WHERE expires_at <= :now")

    def check(self, ip: str) -> Refusal | None:
        """Count one question. Returns why it can't go ahead, or None when it can."""
        client = client_bucket(ip)
        now = int(self._clock())

        with self._engine.connect() as conn:

            def hit(key: str, window: int) -> tuple[int, int]:
                params = {"key": key, "now": now, "expires_at": now + window}
                row = conn.execute(self._hit, params).one()
                return row.hits, row.expires_at - now

            hits, wait = hit(f"ip:{client}:minute", MINUTE)
            if hits > self._per_minute:
                return Refusal(f"That's a lot of questions. Try again in {wait} seconds.", wait)

            hits, wait = hit(f"ip:{client}:day", DAY)
            if hits > self._per_day:
                return Refusal(
                    "You've reached today's limit on this demo. Try again tomorrow, "
                    "or run it yourself from the repo.",
                    wait,
                )

            # Counted last, so one client hammering the route can't spend the budget
            # everyone else shares.
            hits, wait = hit("all:day", DAY)
            if hits == 1:
                conn.execute(self._sweep, {"now": now})
            if hits > self._all_per_day:
                return Refusal(
                    "The demo has used up today's questions for everyone. Try again "
                    "tomorrow, or run it yourself from the repo.",
                    wait,
                )

        return None


def client_ip(headers: Mapping[str, str], peer: str | None) -> str:
    """Vercel overwrites these headers, so a client can't forge them there.

    Behind anything else, the proxy in front has to do the same.
    """
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    return headers.get("x-real-ip") or forwarded or peer or "unknown"


def client_bucket(ip: str) -> str:
    """One IPv6 customer usually holds a whole /64, so that's the unit to count."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return "unknown"
    if address.version == 4:
        return str(address)
    if address.ipv4_mapped:
        return str(address.ipv4_mapped)
    return str(ipaddress.ip_network((address, 64), strict=False).network_address)

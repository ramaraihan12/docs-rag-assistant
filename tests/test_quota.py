import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from docs_rag.quota import DAY, MINUTE, Quota, client_bucket, client_ip


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def engine():
    return create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )


@pytest.fixture
def quota(engine, clock: Clock) -> Quota:
    return Quota(engine, "chunks_quota", per_minute=2, per_day=3, all_per_day=4, clock=clock)


def test_allows_up_to_the_minute_limit_then_refuses(quota: Quota):
    assert quota.check("1.1.1.1") is None
    assert quota.check("1.1.1.1") is None

    refusal = quota.check("1.1.1.1")

    assert refusal is not None
    assert refusal.retry_after == MINUTE
    assert "60 seconds" in refusal.message


def test_the_minute_limit_resets_after_its_window(quota: Quota, clock: Clock):
    quota.check("1.1.1.1")
    quota.check("1.1.1.1")
    assert quota.check("1.1.1.1") is not None

    clock.now += MINUTE

    assert quota.check("1.1.1.1") is None


def test_the_day_limit_outlasts_the_minute_window(quota: Quota, clock: Clock):
    for _ in range(3):
        assert quota.check("1.1.1.1") is None
        clock.now += MINUTE

    refusal = quota.check("1.1.1.1")

    assert refusal is not None
    assert "today's limit" in refusal.message
    assert refusal.retry_after == DAY - 3 * MINUTE


def test_the_daily_total_is_shared_between_clients(quota: Quota):
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"):
        assert quota.check(ip) is None

    refusal = quota.check("5.5.5.5")

    assert refusal is not None
    assert "for everyone" in refusal.message


def test_a_hammering_client_cant_spend_the_shared_total(quota: Quota):
    for _ in range(50):
        quota.check("1.1.1.1")

    assert quota.check("2.2.2.2") is None


def test_two_instances_share_one_count(engine, clock: Clock):
    limits = {"per_minute": 2, "per_day": 9, "all_per_day": 9, "clock": clock}
    first = Quota(engine, "chunks_quota", **limits)
    second = Quota(engine, "chunks_quota", **limits)

    assert first.check("1.1.1.1") is None
    assert second.check("1.1.1.1") is None
    assert first.check("1.1.1.1") is not None


def test_expired_rows_are_swept_when_the_daily_total_rolls_over(quota: Quota, engine, clock: Clock):
    quota.check("1.1.1.1")
    quota.check("2.2.2.2")

    clock.now += DAY
    quota.check("3.3.3.3")

    with engine.connect() as conn:
        keys = conn.execute(text("SELECT key FROM chunks_quota ORDER BY key")).scalars().all()
    assert keys == ["all:day", "ip:3.3.3.3:day", "ip:3.3.3.3:minute"]


def test_client_ip_prefers_the_platform_header():
    headers = {"x-real-ip": "1.1.1.1", "x-forwarded-for": "2.2.2.2, 3.3.3.3"}

    assert client_ip(headers, "4.4.4.4") == "1.1.1.1"


def test_client_ip_falls_back_in_order():
    assert client_ip({"x-forwarded-for": "2.2.2.2, 3.3.3.3"}, "4.4.4.4") == "2.2.2.2"
    assert client_ip({}, "4.4.4.4") == "4.4.4.4"
    assert client_ip({}, None) == "unknown"


def test_ipv6_is_counted_per_64():
    first = client_bucket("2001:db8:1:2:aaaa::1")
    second = client_bucket("2001:db8:1:2:bbbb::2")
    other = client_bucket("2001:db8:1:3::1")

    assert first == second
    assert first != other


def test_ipv4_keeps_its_address_even_when_mapped_into_ipv6():
    assert client_bucket("1.2.3.4") == "1.2.3.4"
    assert client_bucket("::ffff:1.2.3.4") == "1.2.3.4"


def test_anything_that_isnt_an_address_lands_in_one_bucket():
    assert client_bucket("not-an-ip") == "unknown"
    assert client_bucket("") == "unknown"

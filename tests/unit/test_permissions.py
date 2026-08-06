"""Unit tests for permission_is_satisfied — the three-branch RBAC resolver
(direct / legacy-implies / reverse-implication). A regression here is a silent
privilege change, so every branch is pinned."""
import pytest

from core.permissions import LEGACY_PERMISSION_IMPLICATIONS, permission_is_satisfied

pytestmark = pytest.mark.unit


def test_direct_grant():
    assert permission_is_satisfied({"users:read"}, "users:read") is True


def test_missing_grant():
    assert permission_is_satisfied({"users:read"}, "users:delete") is False


def test_empty_permission_set():
    assert permission_is_satisfied(set(), "users:read") is False


class TestLegacyImplication:
    def test_legacy_key_implies_its_members(self):
        # manage_users implies users:delete (and 12 others)
        assert permission_is_satisfied({"manage_users"}, "users:delete") is True

    def test_legacy_key_does_not_imply_unrelated_permission(self):
        # ai_access does not grant user-management permissions
        assert permission_is_satisfied({"ai_access"}, "users:delete") is False


class TestReverseImplication:
    def test_holding_the_full_implied_set_satisfies_the_legacy_key(self):
        implied = set(LEGACY_PERMISSION_IMPLICATIONS["ai_access"])
        assert permission_is_satisfied(implied, "ai_access") is True

    def test_holding_a_partial_set_does_not_satisfy_the_legacy_key(self):
        implied = set(LEGACY_PERMISSION_IMPLICATIONS["ai_access"])
        implied.discard(next(iter(implied)))  # drop one member
        assert permission_is_satisfied(implied, "ai_access") is False

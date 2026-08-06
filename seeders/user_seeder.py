from sqlalchemy.orm import Session
from passlib.context import CryptContext
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uuid
from db.session import SessionLocal
from models.user import User
from models.roles import Role
from models.permissions import Permission
from models.user_roles import UserRole
from models.role_permissions import RolePermission
from dotenv import load_dotenv
from core.permissions import TARGET_PERMISSION_CATALOG

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
load_dotenv()

# The seeded accounts' identities. Overridable, because these are the credentials someone
# signs in with and a hard-coded address in a public repository is both a branding leak and
# a guessable half of a login.
#
# `.local` is deliberate: it is reserved and unroutable, so a default install cannot be
# mistaken for a real mailbox and no verification mail can escape to a domain you do not own.
#
# NOTE FOR EXISTING INSTALLS: the seeder matches on email, so changing this value does not
# rename an account that is already in the database — it creates a second one. If you seeded
# before this changed, either set SUPERADMIN_EMAIL to the address you already have, or update
# the existing row.
DEFAULT_SUPERADMIN_EMAIL = "superadmin@nexarag.local"
DEFAULT_CUSTOMER_EMAIL = "customer@nexarag.local"


def _get_or_create_role(session: Session, name: str) -> Role:
    role = session.query(Role).filter(Role.name == name).first()
    if role:
        return role
    role = Role(id=str(uuid.uuid4()), name=name)
    session.add(role)
    session.flush()
    return role


def _get_or_create_permission(session: Session, name: str, machine_name: str) -> Permission:
    permission = session.query(Permission).filter(Permission.machine_name == machine_name).first()
    if permission:
        return permission
    permission = Permission(id=str(uuid.uuid4()), name=name, machine_name=machine_name)
    session.add(permission)
    session.flush()
    return permission


def _assign_permission(session: Session, role_id: str, permission_id: str) -> None:
    exists = session.query(RolePermission).filter(
        RolePermission.role_id == role_id,
        RolePermission.permission_id == permission_id,
    ).first()
    if not exists:
        session.add(RolePermission(id=str(uuid.uuid4()), role_id=role_id, permission_id=permission_id))


def _assign_role(session: Session, user_id: str, role_id: str) -> None:
    exists = session.query(UserRole).filter(
        UserRole.user_id == user_id,
        UserRole.role_id == role_id,
    ).first()
    if not exists:
        session.add(UserRole(id=str(uuid.uuid4()), user_id=user_id, role_id=role_id))

def seed_database():
    session: Session = SessionLocal()

    try:
        admin_role = _get_or_create_role(session, "Admin")
        customer_role = _get_or_create_role(session, "Customer")

        manage_users_permission = _get_or_create_permission(session, "Manage Users", "manage_users")
        ai_access_permission = _get_or_create_permission(session, "AI Access", "ai_access")
        target_permissions = [
            _get_or_create_permission(session, name, machine_name)
            for name, machine_name in TARGET_PERMISSION_CATALOG
        ]

        # Seed Users
        default_password = os.getenv("SUPERADMIN_PASSWORD")
        if not default_password:
            raise RuntimeError("SUPERADMIN_PASSWORD is required to seed default users")

        # `or`, not os.getenv's default argument: .env.example ships these as blank
        # placeholders, and a key that is present-but-empty makes getenv return "" rather
        # than fall back — which would seed an account with no address at all. This is the
        # same distinction between "set" and "has a value" that broke scripts/bootstrap.sh.
        admin_email = os.getenv("SUPERADMIN_EMAIL", "").strip() or DEFAULT_SUPERADMIN_EMAIL
        customer_email = os.getenv("SEED_CUSTOMER_EMAIL", "").strip() or DEFAULT_CUSTOMER_EMAIL

        admin_user = session.query(User).filter(User.email == admin_email).first()
        if not admin_user:
            admin_user = User(
                id=str(uuid.uuid4()),
                email=admin_email,
                username=admin_email,
                first_name="Nexa",
                last_name="Admin",
                organization="NexaRAG",
                department="dev",
                password=pwd_context.hash(default_password),
                status=1,
            )
            session.add(admin_user)
            session.flush()

        customer_user = session.query(User).filter(User.email == customer_email).first()
        if not customer_user:
            customer_user = User(
                id=str(uuid.uuid4()),
                email=customer_email,
                username=customer_email,
                first_name="Nexa",
                last_name="Customer",
                organization="NexaRAG",
                department="HR",
                password=pwd_context.hash(default_password),
                status=1,
            )
            session.add(customer_user)
            session.flush()

        _assign_role(session, admin_user.id, admin_role.id)
        _assign_role(session, customer_user.id, customer_role.id)

        _assign_permission(session, admin_role.id, manage_users_permission.id)
        _assign_permission(session, admin_role.id, ai_access_permission.id)
        for permission in target_permissions:
            _assign_permission(session, admin_role.id, permission.id)

        _assign_permission(session, customer_role.id, ai_access_permission.id)
        for permission in target_permissions:
            if permission.machine_name in {"chat:ask", "chat:history:read", "chat:feedback:create", "resources:read"}:
                _assign_permission(session, customer_role.id, permission.id)

        # Commit all changes
        session.commit()
        print("Database seeding completed successfully!")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")

    finally:
        session.close()


if __name__ == "__main__":
    seed_database()

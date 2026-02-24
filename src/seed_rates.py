from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rate_service import DEFAULT_POLICY_ID, seed_default_rates


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create and seed database rate tables for the rating service."
    )
    parser.add_argument(
        "--policy-id",
        default=DEFAULT_POLICY_ID,
        help="Policy id to seed rates for.",
    )
    args = parser.parse_args()

    seed_default_rates(policy_id=str(args.policy_id))
    print(f"Seeded rate tables for policy_id='{args.policy_id}'")


if __name__ == "__main__":
    main()

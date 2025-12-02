#!/usr/bin/env python3
"""
Check email configuration in the database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.db import get_db, connect_db, EMAIL_CONFIG_COLLECTION


def check_email_config():
    """Check email configuration in database"""
    try:
        # Connect to database
        connect_db()
        db = get_db()
        
        # Get all email configs
        configs = list(db[EMAIL_CONFIG_COLLECTION].find().sort("updated_at", -1))
        
        if not configs:
            print("❌ No email configuration found in database.")
            return
        
        print(f"📧 Found {len(configs)} email configuration(s) in database:\n")
        
        for i, config in enumerate(configs, 1):
            print(f"Configuration #{i}:")
            print(f"   ID: {config.get('_id')}")
            print(f"   SMTP Host: {config.get('smtp_host', 'N/A')}")
            print(f"   SMTP Port: {config.get('smtp_port', 'N/A')}")
            print(f"   SMTP User: {config.get('smtp_user', 'N/A')}")
            print(f"   SMTP Password: {'*' * len(config.get('smtp_password', '')) if config.get('smtp_password') else 'Not Set'}")
            print(f"   SMTP From: {config.get('smtp_from', 'N/A')}")
            print(f"   Enabled: {config.get('enabled', False)}")
            print(f"   Created At: {config.get('created_at', 'N/A')}")
            print(f"   Updated At: {config.get('updated_at', 'N/A')}")
            print(f"   Created By: {config.get('created_by', 'N/A')}")
            print()
        
        # Check the latest config (the one that should be used)
        latest_config = configs[0]
        
        print("🔍 Status Check:")
        if latest_config.get('enabled'):
            print("   ✅ Email is ENABLED")
        else:
            print("   ❌ Email is DISABLED")
        
        if latest_config.get('smtp_user'):
            print(f"   ✅ SMTP User is set: {latest_config.get('smtp_user')}")
        else:
            print("   ❌ SMTP User is NOT set")
        
        if latest_config.get('smtp_password'):
            print(f"   ✅ SMTP Password is set (length: {len(latest_config.get('smtp_password'))})")
        else:
            print("   ❌ SMTP Password is NOT set")
        
        if latest_config.get('enabled') and latest_config.get('smtp_user') and latest_config.get('smtp_password'):
            print("\n✅ Email configuration is COMPLETE and should work!")
        else:
            print("\n❌ Email configuration is INCOMPLETE. Please run init_email_config.py to fix it.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    check_email_config()


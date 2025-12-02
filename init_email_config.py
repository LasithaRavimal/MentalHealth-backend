#!/usr/bin/env python3
"""
Simple script to initialize email configuration in the database.
This script can be run to set up email configuration without using the API.

Usage:
    python init_email_config.py

Or with environment variables:
    EMAIL_ENABLED=true SMTP_USER=your_email@gmail.com SMTP_PASSWORD=your_app_password python init_email_config.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.db import get_db, connect_db, EMAIL_CONFIG_COLLECTION


def init_email_config():
    """Initialize email configuration in database"""
    try:
        # Connect to database
        connect_db()
        db = get_db()
        
        # Check if config already exists
        existing_config = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
        
        if existing_config:
            print("⚠️  Email configuration already exists in database.")
            print(f"   Config ID: {existing_config['_id']}")
            print(f"   SMTP User: {existing_config.get('smtp_user', 'N/A')}")
            print(f"   Enabled: {existing_config.get('enabled', False)}")
            print(f"   Updated: {existing_config.get('updated_at', 'N/A')}")
            
            response = input("\nDo you want to update it? (y/N): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        else:
            print("📧 Email configuration not found. Creating new configuration...\n")
        
        # Get configuration from user or environment variables
        print("Enter email configuration:")
        print("(Press Enter to use default or environment variable values)\n")
        
        # Get SMTP settings
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        
        # Prompt for values if not in environment
        if not smtp_user:
            smtp_user = input("SMTP User (email): ").strip()
        else:
            print(f"SMTP User: {smtp_user} (from environment)")
        
        if not smtp_password:
            smtp_password = input("SMTP Password/App Password: ").strip()
        else:
            print(f"SMTP Password: {'*' * len(smtp_password)} (from environment)")
        
        smtp_host_input = input(f"SMTP Host [{smtp_host}]: ").strip()
        if smtp_host_input:
            smtp_host = smtp_host_input
        
        smtp_port_input = input(f"SMTP Port [{smtp_port}]: ").strip()
        if smtp_port_input:
            smtp_port = int(smtp_port_input)
        
        enabled_input = input(f"Enable email sending? (y/N) [{enabled}]: ").strip().lower()
        if enabled_input:
            enabled = enabled_input == 'y'
        
        if not smtp_user or not smtp_password:
            print("❌ Error: SMTP User and Password are required!")
            return
        
        # Create/update configuration
        now = datetime.utcnow()
        config_doc = {
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
            "smtp_from": smtp_user,
            "enabled": enabled,
            "updated_at": now,
            "created_by": "script_init"
        }
        
        if existing_config:
            # Update existing
            db[EMAIL_CONFIG_COLLECTION].update_one(
                {"_id": existing_config["_id"]},
                {"$set": config_doc}
            )
            print(f"\n✅ Email configuration updated successfully!")
            print(f"   Config ID: {existing_config['_id']}")
        else:
            # Create new
            config_doc["created_at"] = now
            result = db[EMAIL_CONFIG_COLLECTION].insert_one(config_doc)
            print(f"\n✅ Email configuration created successfully!")
            print(f"   Config ID: {result.inserted_id}")
        
        print(f"\n📋 Configuration Summary:")
        print(f"   SMTP Host: {smtp_host}")
        print(f"   SMTP Port: {smtp_port}")
        print(f"   SMTP User: {smtp_user}")
        print(f"   SMTP From: {smtp_user}")
        print(f"   Enabled: {enabled}")
        print(f"\n💡 Restart your backend server for changes to take effect.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    init_email_config()


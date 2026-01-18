import logging
from typing import Dict, Any, Optional, Tuple
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import settings, refresh_email_config

logger = logging.getLogger(__name__)


async def send_email(to_email: str, subject: str, body_html: str, body_text: Optional[str] = None) -> bool:
    """
    Send an email using SMTP.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body_html: HTML email body
        body_text: Plain text email body (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Refresh email config from database before sending
    refresh_email_config()
    
    if not settings.EMAIL_ENABLED:
        logger.info(f"Email sending is disabled (EMAIL_ENABLED=False). Would send to {to_email}: {subject}")
        return False
    
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            f"SMTP credentials not configured. Email sending skipped. "
            f"SMTP_USER: {'Set' if settings.SMTP_USER else 'Not Set'}, "
            f"SMTP_PASSWORD: {'Set' if settings.SMTP_PASSWORD else 'Not Set'}. "
            f"Please configure email settings via admin API or environment variables."
        )
        return False
    
    try:
        # Get current config values (may have been refreshed from DB)
        smtp_host = settings.SMTP_HOST
        smtp_port = settings.SMTP_PORT
        smtp_user = settings.SMTP_USER
        smtp_password = settings.SMTP_PASSWORD
        smtp_from = settings.SMTP_FROM or smtp_user
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        
        # Format From address - if SMTP_FROM doesn't include a name, add one
        from_address = smtp_from
        if from_address and "<" not in from_address:
            # Add display name if not already present
            from_address = f"M_Track System <{from_address}>"
        message["From"] = from_address
        
        message["To"] = to_email
        
        # Add plain text part if provided
        if body_text:
            text_part = MIMEText(body_text, "plain")
            message.attach(text_part)
        
        # Add HTML part
        html_part = MIMEText(body_html, "html")
        message.attach(html_part)
        
        # Send email
        # For port 587 (Gmail), use STARTTLS (upgrade plain connection to TLS)
        # For port 465, use direct SSL/TLS connection
        if smtp_port == 465:
            # Port 465 uses direct SSL/TLS (implicit TLS)
            await aiosmtplib.send(
                message,
                hostname=smtp_host,
                port=smtp_port,
                username=smtp_user,
                password=smtp_password,
                use_tls=True,  # Direct SSL/TLS
            )
        else:
            # Port 587 and others use STARTTLS (explicit TLS upgrade)
            # Use the helper function with start_tls=True parameter
            await aiosmtplib.send(
                message,
                hostname=smtp_host,
                port=smtp_port,
                username=smtp_user,
                password=smtp_password,
                use_tls=False,  # Don't use direct TLS
                start_tls=True,  # Use STARTTLS instead
            )
        
        logger.info(f"Email sent successfully to {to_email} from {smtp_from}: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False


def create_stress_alert_email_body(user_email: str, prediction_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Create email body for stress alert.
    
    Args:
        user_email: User's email address
        prediction_data: Prediction data containing stress/depression levels and explanations
        
    Returns:
        Tuple of (html_body, text_body)
    """
    stress_level = prediction_data.get("stress_level", "Unknown")
    depression_level = prediction_data.get("depression_level", "Unknown")
    explanations = prediction_data.get("explanations", [])
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #1DB954; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
            .content {{ background-color: #f9f9f9; padding: 20px; border-radius: 0 0 5px 5px; }}
            .alert {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
            .explanation {{ background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>M_Track Mood Alert</h1>
            </div>
            <div class="content">
                <h2>Important: Elevated Stress Detected</h2>
                <div class="alert">
                    <p><strong>Your last listening session indicates elevated stress levels.</strong></p>
                    <p>Stress Level: <strong>{stress_level}</strong></p>
                    <p>Depression Level: <strong>{depression_level}</strong></p>
                </div>
                
                <h3>Session Insights:</h3>
                <ul>
                    {''.join([f'<li class="explanation">{exp}</li>' for exp in explanations[:5]])}
                </ul>
                
                <h3>Recommendations:</h3>
                <ul>
                    <li>Consider listening to calming or relaxing music</li>
                    <li>Take a short break from music and practice deep breathing</li>
                    <li>Try engaging in light physical activity</li>
                    <li>Consider speaking with a healthcare professional if stress persists</li>
                </ul>
                
                <p>Remember, M_Track is a tool to help you understand your listening patterns. 
                If you're experiencing ongoing stress or depression, please seek professional help.</p>
                
                <div class="footer">
                    <p>This is an automated alert from M_Track - AI-Based Music Behavior Analysis Platform</p>
                    <p>You received this email because your recent session indicated elevated stress levels.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
M_Track Mood Alert

IMPORTANT: Elevated Stress Detected

Your last listening session indicates elevated stress levels.

Stress Level: {stress_level}
Depression Level: {depression_level}

Session Insights:
{chr(10).join(['- ' + exp for exp in explanations[:5]])}

Recommendations:
- Consider listening to calming or relaxing music
- Take a short break from music and practice deep breathing
- Try engaging in light physical activity
- Consider speaking with a healthcare professional if stress persists

Remember, M_Track is a tool to help you understand your listening patterns. 
If you're experiencing ongoing stress or depression, please seek professional help.

---
This is an automated alert from M_Track - AI-Based Music Behavior Analysis Platform
You received this email because your recent session indicated elevated stress levels.
    """
    
    return html_body, text_body


def create_depression_alert_email_body(user_email: str, prediction_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Create email body for depression alert.
    
    Args:
        user_email: User's email address
        prediction_data: Prediction data containing stress/depression levels and explanations
        
    Returns:
        Tuple of (html_body, text_body)
    """
    stress_level = prediction_data.get("stress_level", "Unknown")
    depression_level = prediction_data.get("depression_level", "Unknown")
    explanations = prediction_data.get("explanations", [])
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #1DB954; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
            .content {{ background-color: #f9f9f9; padding: 20px; border-radius: 0 0 5px 5px; }}
            .alert {{ background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; }}
            .explanation {{ background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            .resources {{ background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>M_Track Mood Alert</h1>
            </div>
            <div class="content">
                <h2>Important: Elevated Depression Detected</h2>
                <div class="alert">
                    <p><strong>Your last listening session indicates elevated depression levels.</strong></p>
                    <p>Stress Level: <strong>{stress_level}</strong></p>
                    <p>Depression Level: <strong>{depression_level}</strong></p>
                </div>
                
                <h3>Session Insights:</h3>
                <ul>
                    {''.join([f'<li class="explanation">{exp}</li>' for exp in explanations[:5]])}
                </ul>
                
                <h3>Recommendations:</h3>
                <ul>
                    <li>Consider listening to uplifting or energetic music</li>
                    <li>Engage in activities you enjoy</li>
                    <li>Connect with friends or family</li>
                    <li>Consider professional mental health support</li>
                </ul>
                
                <div class="resources">
                    <h4>If you need immediate help:</h4>
                    <p>If you're experiencing thoughts of self-harm or suicide, please reach out to:</p>
                    <ul>
                        <li>National Suicide Prevention Lifeline: 988 (US)</li>
                        <li>Crisis Text Line: Text HOME to 741741</li>
                        <li>Or contact your local emergency services</li>
                    </ul>
                </div>
                
                <p>Remember, M_Track is a tool to help you understand your listening patterns. 
                If you're experiencing ongoing depression, please seek professional help.</p>
                
                <div class="footer">
                    <p>This is an automated alert from M_Track - AI-Based Music Behavior Analysis Platform</p>
                    <p>You received this email because your recent session indicated elevated depression levels.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
M_Track Mood Alert

IMPORTANT: Elevated Depression Detected

Your last listening session indicates elevated depression levels.

Stress Level: {stress_level}
Depression Level: {depression_level}

Session Insights:
{chr(10).join(['- ' + exp for exp in explanations[:5]])}

Recommendations:
- Consider listening to uplifting or energetic music
- Engage in activities you enjoy
- Connect with friends or family
- Consider professional mental health support

If you need immediate help:
If you're experiencing thoughts of self-harm or suicide, please reach out to:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- Or contact your local emergency services

Remember, M_Track is a tool to help you understand your listening patterns. 
If you're experiencing ongoing depression, please seek professional help.

---
This is an automated alert from M_Track - AI-Based Music Behavior Analysis Platform
You received this email because your recent session indicated elevated depression levels.
    """
    
    return html_body, text_body


async def send_stress_alert(user_email: str, prediction_data: Dict[str, Any]) -> bool:
    """
    Send stress alert email to user.
    
    Args:
        user_email: User's email address
        prediction_data: Prediction data containing stress/depression levels
        
    Returns:
        True if email sent successfully, False otherwise
    """
    subject = "M_Track Mood Alert - Elevated Stress Detected"
    html_body, text_body = create_stress_alert_email_body(user_email, prediction_data)
    
    return await send_email(user_email, subject, html_body, text_body)


async def send_depression_alert(user_email: str, prediction_data: Dict[str, Any]) -> bool:
    """
    Send depression alert email to user.
    
    Args:
        user_email: User's email address
        prediction_data: Prediction data containing stress/depression levels
        
    Returns:
        True if email sent successfully, False otherwise
    """
    subject = "M_Track Mood Alert - Elevated Depression Detected"
    html_body, text_body = create_depression_alert_email_body(user_email, prediction_data)
    
    return await send_email(user_email, subject, html_body, text_body)


def create_welcome_email_body(user_email: str, user_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Create welcome email body for new user registration.
    """
    display_name = user_name or user_email.split('@')[0]

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #1DB954; color: white; padding: 30px; text-align: center; border-radius: 5px 5px 0 0; }}
            .content {{ background-color: #f9f9f9; padding: 30px; border-radius: 0 0 5px 5px; }}
            .feature {{ background-color: white; border-left: 4px solid #1DB954; padding: 15px; margin: 15px 0; }}
            .button {{ display: inline-block; background-color: #1DB954; color: white; padding: 12px 30px; text-decoration: none; border-radius: 9999px; font-weight: 700; margin: 20px 0; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Welcome to M_Track</h1>
                <p>AI-Based Mental Health Detection Platform</p>
            </div>

            <div class="content">
                <h2>Hello {display_name},</h2>

                <p>
                    Welcome to <strong>M_Track</strong> – an AI-powered mental health detection system
                    designed to identify early signs of <strong>stress and depression</strong>
                    using multiple behavioral and biometric data sources.
                </p>

                <h3>Detection Methods Used in M_Track:</h3>

                <div class="feature">
                    <strong>🎵 Music Behavior Analysis</strong><br/>
                    Analyze listening patterns such as skips, repeats, duration, and mood-based categories.
                </div>

                <div class="feature">
                    <strong>🙂 Facial Expression Analysis</strong><br/>
                    Detect emotional cues from facial movements and expressions.
                </div>

                <div class="feature">
                    <strong>🎤 Voice Signal Analysis</strong><br/>
                    Identify emotional stress indicators from voice tone and speech patterns.
                </div>

                <div class="feature">
                    <strong>🧠 EEG Signal Analysis</strong><br/>
                    Analyze brainwave signals to support mental state detection.
                </div>

                <h3>How It Works:</h3>
                <ol>
                    <li>You interact naturally with the system (music, face, voice, EEG)</li>
                    <li>Behavioral data is collected during sessions</li>
                    <li>AI models analyze patterns</li>
                    <li>Stress & depression levels are predicted</li>
                </ol>

                <div style="text-align: center;">
                    <a href="http://localhost:5173/home" class="button">Start Using M_Track</a>
                </div>

                <p>
                    <strong>Privacy Notice:</strong> Your data is processed securely and used only for
                    research and mental well-being insights.
                </p>

                <p>Thank you for joining M_Track.</p>
                <p><strong>– The M_Track Team</strong></p>

                <div class="footer">
                    <p>This is an automated welcome email from M_Track</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    text_body = f"""
Welcome to M_Track

Hello {display_name},

Welcome to M_Track – an AI-based mental health detection platform.

M_Track analyzes mental well-being using:
- Music listening behavior
- Facial expressions
- Voice signals
- EEG brainwave data

How it works:
1. You interact naturally with the system
2. Behavioral and biometric data is collected
3. AI models analyze the data
4. Stress and depression levels are predicted

Your privacy is respected and data is handled securely.

Thank you for joining M_Track.
– The M_Track Team
"""

    return html_body, text_body



async def send_welcome_email(user_email: str, user_name: Optional[str] = None) -> bool:
    """
    Send welcome email to newly registered user.
    
    Args:
        user_email: User's registered email address
        user_name: User's name (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    subject = "Welcome to M_Track - Get Started with AI-Powered Music Analysis"
    html_body, text_body = create_welcome_email_body(user_email, user_name)
    
    return await send_email(user_email, subject, html_body, text_body)


def create_logout_email_body(user_email: str, prediction_data: Dict[str, Any]) -> Tuple[str, str]:
    """Create email body for logout notification with latest session prediction"""
    stress_level = prediction_data.get("stress_level", "Unknown")
    depression_level = prediction_data.get("depression_level", "Unknown")
    explanations = prediction_data.get("explanations", [])
    
    stress_class = stress_level.lower() if stress_level else "unknown"
    depression_class = depression_level.lower() if depression_level else "unknown"
    
    # Build explanations HTML separately to avoid f-string nesting issues
    explanations_html = ""
    if explanations:
        explanations_items = ''.join([f'<li class="explanation">{exp}</li>' for exp in explanations[:5]])
        explanations_html = f'<h3>Session Insights:</h3><ul>{explanations_items}</ul>'
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #1DB954; color: white; padding: 30px; text-align: center; border-radius: 5px 5px 0 0; }}
            .content {{ background-color: #f9f9f9; padding: 30px; border-radius: 0 0 5px 5px; }}
            .prediction {{ background-color: white; border-left: 4px solid #1DB954; padding: 15px; margin: 15px 0; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            .stress-high {{ color: #dc2626; font-weight: bold; }}
            .stress-moderate {{ color: #d97706; font-weight: bold; }}
            .stress-low {{ color: #16a34a; font-weight: bold; }}
            .explanation {{ margin: 8px 0; padding-left: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>M_Track Session Summary</h1>
                <p>Your Latest Stress & Depression Levels</p>
            </div>
            <div class="content">
                <h2>Hi there,</h2>
                <p>You've logged out from M_Track. Here's a summary of your latest listening session:</p>
                
                <div class="prediction">
                    <h3>Session Results:</h3>
                    <p><strong>Stress Level:</strong> <span class="stress-{stress_class}">{stress_level}</span></p>
                    <p><strong>Depression Level:</strong> <span class="stress-{depression_class}">{depression_level}</span></p>
                </div>
                
                {explanations_html}
                
                <p>Thank you for using M_Track. See you next time!</p>
                
                <div class="footer">
                    <p>This is an automated email from M_Track - AI-Based Music Behavior Analysis Platform</p>
                    <p>You received this email because you logged out from your account.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
M_Track Session Summary

Hi there,

You've logged out from M_Track. Here's a summary of your latest listening session:

Session Results:
Stress Level: {stress_level}
Depression Level: {depression_level}

{f'Session Insights:{chr(10)}{chr(10).join(["- " + exp for exp in explanations[:5]])}' if explanations else ''}

Thank you for using M_Track. See you next time!

---
This is an automated email from M_Track - AI-Based Music Behavior Analysis Platform
You received this email because you logged out from your account.
    """
    
    return html_body, text_body


async def send_logout_email(user_email: str, prediction_data: Dict[str, Any]) -> bool:
    """Send logout notification email with latest session prediction"""
    subject = "M_Track Session Summary - Your Latest Stress & Depression Levels"
    html_body, text_body = create_logout_email_body(user_email, prediction_data)
    
    return await send_email(user_email, subject, html_body, text_body)


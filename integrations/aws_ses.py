import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from schemas.email_schema import EmailTemplate

logger = logging.getLogger(__name__)


def send_email(recipient_email: str, email_template: EmailTemplate):
    """Send email via SMTP. Falls back to logging if SMTP is not configured."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
    from_name = os.getenv("SMTP_FROM_NAME", "NexaRAG").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    if not smtp_host or not from_email:
        logger.info("SMTP not configured — logging email instead.")
        logger.info("EMAIL — To: %s | Subject: %s", recipient_email, email_template.subject)
        logger.info("EMAIL BODY:\n%s", email_template.body)
        return {"status": True, "message": "Email logged (SMTP not configured)"}

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = email_template.subject
        msg["From"] = f"{from_name} <{from_email}>"
        msg["To"] = recipient_email

        msg.attach(MIMEText(email_template.body, "html"))

        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)

        if smtp_username and smtp_password:
            server.login(smtp_username, smtp_password)

        server.sendmail(from_email, recipient_email, msg.as_string())
        server.quit()

        logger.info("Email sent to %s — Subject: %s", recipient_email, email_template.subject)
        return {"status": True, "message": "Email sent successfully"}
    except Exception as e:
        logger.error("Failed to send email to %s: %s", recipient_email, str(e))
        return {"status": False, "message": f"Failed to send email: {str(e)}"}

import smtplib
import os
import logging
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)


def get_latest_report(output_dir):
    """Find the most recent report file in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    reports = list(output_path.glob("support_rep_report_*.md"))
    if not reports:
        return None

    return max(reports, key=lambda p: p.stat().st_mtime)


def markdown_to_text(markdown_text):
    """Convert markdown to plain text for email body."""
    text = markdown_text

    text = text.replace("## ", "\n")
    text = text.replace("### ", "\n")

    text = text.replace("**", "")
    text = text.replace("__", "")

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    result = "\n".join(cleaned_lines)
    result = result.replace("-\n", "- ")
    result = result.replace("\n- ", "\n  - ")

    return result


def send_report_email(config, report_body=None, output_dir="output/"):
    """
    Send report email via SMTP.

    Args:
        config (dict): Configuration with email settings
        report_body (str): Report content to send. If None, uses most recent report file.
        output_dir (str): Path to output directory (used if report_body is None)

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    email_config = config.get("email", {})

    if not email_config.get("enabled", False):
        logger.warning("Email not enabled in config")
        return False

    smtp_host = email_config.get("smtp_host")
    smtp_port = email_config.get("smtp_port", 587)
    use_tls = email_config.get("use_tls", True)
    from_email = email_config.get("from")
    to_emails = email_config.get("to", [])
    subject_prefix = email_config.get("subject_prefix", "Support Call QA Report")

    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_host, from_email, to_emails, smtp_username, smtp_password]):
        logger.error("Missing email configuration. Check config.json and .env")
        logger.error(f"  smtp_host: {'set' if smtp_host else 'MISSING'}")
        logger.error(f"  from_email: {'set' if from_email else 'MISSING'}")
        logger.error(f"  to_emails: {'set' if to_emails else 'MISSING'}")
        logger.error(f"  SMTP_USERNAME: {'set' if smtp_username else 'MISSING'}")
        logger.error(f"  SMTP_PASSWORD: {'set' if smtp_password else 'MISSING'}")
        return False

    if report_body is None:
        report_file = get_latest_report(output_dir)
        if report_file is None:
            logger.error("No report file found in output directory")
            return False

        logger.info(f"Using report file: {report_file.name}")
        with open(report_file, "r") as f:
            report_body = f.read()

    body_text = markdown_to_text(report_body)

    subject = f"{subject_prefix} - {datetime.now().strftime('%Y-%m-%d')}"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = ", ".join(to_emails)

        part = MIMEText(body_text, "plain")
        msg.attach(part)

        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)

        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_emails, msg.as_string())
        server.quit()

        logger.info(f"Email sent successfully to: {', '.join(to_emails)}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

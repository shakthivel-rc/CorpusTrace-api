def login_email_template(first_name: str, verify_url: str):
    email_body = f"""<!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Welcome to CorpusTrace</title>
        </head>
        <body style="margin:0; padding:0; background-color:#f4f4f4;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" bgcolor="#f4f4f4">
            <tr>
            <td align="center" style="padding: 40px 0;">
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="background-color:#ffffff; border-radius:6px; padding: 40px; font-family:Arial, sans-serif; color:#333333;">
                <tr>
                    <td align="left" style="font-size: 18px; line-height: 28px;">
                    <p style="margin: 0 0 20px;">Hi {first_name},</p>
                    <p style="margin: 0 0 20px;">
                        Thank you for signing up with <strong>CorpusTrace</strong>! We are excited to have you on board.
                    </p>
                    <p style="margin: 0 0 20px;">
                        Your account has been successfully created.
                    </p>
                    <p style="margin: 0 0 40px;">
                        <a href="{verify_url}" style="display:inline-block; padding:12px 24px; background-color:#007BFF; color:#ffffff; text-decoration:none; border-radius:4px; font-weight:bold;">Verify your account</a>
                    </p>
                    <p style="margin: 0;">
                        Welcome aboard!<br>
                        The CorpusTrace Team
                    </p>
                    </td>
                </tr>
                </table>
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="margin-top:20px;">
                <tr>
                    <td align="center" style="font-size:12px; color:#888888; font-family:Arial, sans-serif;">
                    &copy; 2026 SHAKTHIVEL RAVICHANDRAN. CorpusTrace is open source under the Apache License 2.0.
                    </td>
                </tr>
                </table>
            </td>
            </tr>
        </table>
        </body>
        </html>
    """
    return email_body

def otp_email_template(first_name: str, otp_code: str):
    email_body = f"""<!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Your Verification Code - CorpusTrace</title>
        </head>
        <body style="margin:0; padding:0; background-color:#f4f4f4;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" bgcolor="#f4f4f4">
            <tr>
            <td align="center" style="padding: 40px 0;">
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="background-color:#ffffff; border-radius:6px; padding: 40px; font-family:Arial, sans-serif; color:#333333;">
                <tr>
                    <td align="left" style="font-size: 18px; line-height: 28px;">
                    <p style="margin: 0 0 20px;">Hi {first_name},</p>
                    <p style="margin: 0 0 20px;">
                        Thank you for registering with <strong>CorpusTrace</strong>! To verify your email address, please use the following code:
                    </p>
                    <p style="margin: 0 0 10px; text-align: center;">
                        <span style="display:inline-block; padding:16px 32px; background-color:#f0f4ff; border: 2px dashed #007BFF; border-radius:8px; font-size:32px; font-weight:bold; letter-spacing:8px; color:#007BFF;">{otp_code}</span>
                    </p>
                    <p style="margin: 20px 0; text-align: center; font-size: 14px; color: #888888;">
                        This code will expire in <strong>10 minutes</strong>.
                    </p>
                    <p style="margin: 0 0 20px;">
                        If you didn't create an account with CorpusTrace, you can safely ignore this email.
                    </p>
                    <p style="margin: 0;">
                        Welcome aboard!<br>
                        The CorpusTrace Team
                    </p>
                    </td>
                </tr>
                </table>
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="margin-top:20px;">
                <tr>
                    <td align="center" style="font-size:12px; color:#888888; font-family:Arial, sans-serif;">
                    &copy; 2026 SHAKTHIVEL RAVICHANDRAN. CorpusTrace is open source under the Apache License 2.0.
                    </td>
                </tr>
                </table>
            </td>
            </tr>
        </table>
        </body>
        </html>
    """
    return email_body

def reset_password_email_template(first_name: str, reset_url: str):
    email_body = f"""<!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Reset Your Password - CorpusTrace</title>
        </head>
        <body style="margin:0; padding:0; background-color:#f4f4f4;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" bgcolor="#f4f4f4">
            <tr>
            <td align="center" style="padding: 40px 0;">
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="background-color:#ffffff; border-radius:6px; padding: 40px; font-family:Arial, sans-serif; color:#333333;">
                <tr>
                    <td align="left" style="font-size: 18px; line-height: 28px;">
                    <p style="margin: 0 0 20px;">Hi {first_name},</p>
                    <p style="margin: 0 0 20px;">
                        We received a request to reset your password for your <strong>CorpusTrace</strong> account.
                    </p>
                    <p style="margin: 0 0 20px;">
                        If you made this request, you can reset your password by clicking the button below:
                    </p>
                    <p style="margin: 0 0 40px;">
                        <a href="{reset_url}" style="display:inline-block; padding:12px 24px; background-color:#007BFF; color:#ffffff; text-decoration:none; border-radius:4px; font-weight:bold;">Reset Password</a>
                    </p>
                    <p style="margin: 0 0 20px;">
                        If you didn't request a password reset, you can safely ignore this email. Your password will not be changed.
                    </p>
                    </td>
                </tr>
                </table>
                <table cellpadding="0" cellspacing="0" border="0" width="600" style="margin-top:20px;">
                <tr>
                    <td align="center" style="font-size:12px; color:#888888; font-family:Arial, sans-serif;">
                    &copy; 2026 SHAKTHIVEL RAVICHANDRAN. CorpusTrace is open source under the Apache License 2.0.
                    </td>
                </tr>
                </table>
            </td>
            </tr>
        </table>
        </body>
        </html>
    """
    return email_body

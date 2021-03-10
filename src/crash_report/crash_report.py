import os
import pickle
import base64
from apiclient import errors
from email.mime.text import MIMEText

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _get_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    token_pickle = os.path.join(CUR_DIR, "token.pickle")
    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(CUR_DIR, "credentials.json"), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_pickle, 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)


def _create_message(sender, to, subject, body):
    """Create a message for an email.

    Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

    Returns:
    An object containing a base64url encoded email object.
    """
    message = MIMEText(body, "html")
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode()}


def _send_message(service, user_id, message):
    """Send an email message.

    Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

    Returns:
    Sent Message.
    """
    try:
        message = (service.users().messages().send(userId=user_id,
                                                   body=message).execute())
        return message
    except errors.HttpError as error:
        print(f"An error occurred: {error}")


def send_mail(to, subject, body):
    """Reports a crash by e-mail to `to` with subject `subject` and body `body`

    Args:
        to (str): Adress e-mail of the recipient.
        subject (str): Subject of the e-mail.
        body (str): Body of the e-mail.
    """
    service = _get_service()
    msg = _create_message("", to, subject, body)
    _send_message(service, to, msg)

""" slack api
# TODO : 토큰을 GitHub에 올리면 토큰이 자동으로 재생성되므로 Github에 올리면 안됩니다.
# TODO : secrets.json 파일을 생성하세요.
# TODO : json 파일 내에서 세팅을 설정하세요.
    - TOKEN (required), https://api.slack.com/apps
    - CHANNEL_ID (required)
    - USER_NAME
    - COLOR
    - EMOJI
"""

import os
import json
import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


USER_NAME = "anonymous"
COLOR = "#ffffff"
EMOJI = ":smile:"
CHANNEL_ID = None


def get_slack_client(args):
    global USER_NAME, COLOR, EMOJI, CHANNEL_ID
    file_name = "secrets.json"
    secret_key_path = os.path.join(args.data_path, "keys", file_name)

    assert os.path.exists(secret_key_path), f"{secret_key_path} 파일이 존재하지 않습니다."

    with open(secret_key_path, "r") as secret_file:
        secrets = json.loads(secret_file.read())

    token = secrets["SLACK"]["TOKEN"]
    CHANNEL_ID = secrets["SLACK"]["CHANNEL_ID"]
    client = WebClient(token)

    USER_NAME = secrets["SLACK"]["USER_NAME"]
    COLOR = secrets["SLACK"]["COLOR"]
    EMOJI = secrets["SLACK"]["EMOJI"]

    return client


def get_format_datas(args, run_type, eval_results, use_pororo=False):
    pretext = f"Module: {run_type} {EMOJI}"
    author_name = f"전략: {args.strategy} | 모델: {args.model.reader_name} | PORORO: {use_pororo}"
    title = f"별칭: {args.alias}"
    text = f"*EM*: {eval_results['exact_match']}\n *F1*: {eval_results['f1']}"
    timestamp = int(datetime.datetime.now().timestamp() // 1)

    attachments = [
        {
            "fallback": "Plain-text summary of the attachment.",
            "color": COLOR,
            "pretext": pretext,
            "author_name": author_name,
            "title": title,
            "text": text,
            "footer": f"{USER_NAME}",
            "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
            "ts": timestamp,
        }
    ]

    return attachments


def report_reader_to_slack(args, run_type, eval_results, use_pororo=False):
    """report_reader_to_slack.
    Args:
        run_type: [run_mrc.py or run.py], 모듈 구분을 위한 인자
        eval_results: dict, {'exact_match': '00.00%', 'f1': '00.00%'}
    """

    client = get_slack_client(args)

    attachments = get_format_datas(args, run_type, eval_results, use_pororo)

    try:
        result = client.chat_postMessage(channel=CHANNEL_ID, attachments=attachments)
        print(result)
    except SlackApiError as e:
        print("Error uploading file: {}".format(e))


def report_retriever_to_slack(args, fig):
    """retriever 결과를 slack 으로 전송합니다.
    Args:
        fig : plt.Figure, retriever 성능을 비교한 figure
    """
    client = get_slack_client(args)

    file_path = "./temp.png"
    fig.savefig(file_path)

    try:
        result = client.files_upload(
            file=file_path,
            channels=CHANNEL_ID,
            filename="fig_image.png",
            initial_comment="Retrieval Results! :smile:",
            title="_".join([USER_NAME, "Retrievers Results"]),
        )
        print(result)
    except SlackApiError as e:
        print("Error uploading file: {}".format(e))
    finally:
        os.remove(file_path)

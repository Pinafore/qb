import click
import os
from slackclient import SlackClient

def send_message(channel, message, username):
    slack_token = os.environ.get('SLACK_API_TOKEN')
    client = SlackClient(slack_token)
    client.api_call(
        'chat.postMessage',
        channel=channel, text=message, username=username
    )


@click.command()
@click.option('--bot_name', default='Slurm')
@click.argument('channel')
@click.argument('message')
def main(bot_name, channel, message):
    send_message(channel, message, bot_name)


if __name__ == '__main__':
    main()


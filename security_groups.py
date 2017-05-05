#!/usr/bin/env python3

import subprocess
import json
import os
import hcl


def api(command, parse=True):
    response = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    if parse:
        return json.loads(response.stdout.decode('utf8'))


def get_spot_ids():
    with open('terraform.tfstate') as f:
        state = hcl.load(f)

    resources = state['modules'][0]['resources']

    if 'aws_spot_instance_request.qanta' in resources:
        return [resources['aws_spot_instance_request.qanta']['primary']['id']]
    elif 'aws_spot_instance_request.qanta.0' in resources:
        instances = [r for r in resources if 'aws_spot_instance_request.qanta' in r]
        return [resources[r]['primary']['id'] for r in instances]
    else:
        raise ValueError('No matching instances found')


def get_instance_id(spot_id):
    response = api([
        'aws', 'ec2',
        'describe-spot-instance-requests',
        '--spot-instance-request-ids', spot_id
    ])
    return response['SpotInstanceRequests'][0]['InstanceId']


def get_current_security_groups(instance_id):
    response = api([
        'aws', 'ec2',
        'describe-instance-attribute',
        '--attribute', 'groupSet',
        '--instance-id', instance_id
    ])
    return [g['GroupId'] for g in response['Groups']]


def attach_security_group(instance_id, sids):
    security_groups = get_current_security_groups(instance_id)
    security_groups.extend(sids)
    api([
        'aws', 'ec2',
        'modify-instance-attribute',
        '--instance-id', instance_id,
        '--groups'
    ] + security_groups, parse=False)


if __name__ == '__main__':
    security_groups = os.environ.get('QB_SECURITY_GROUPS')
    if security_groups is not None:
        print('Adding these security groups:', security_groups)
        security_groups = security_groups.split(',')
        spot_ids = get_spot_ids()
        for spot_id in spot_ids:
            instance_id = get_instance_id(spot_id)
            attach_security_group(instance_id, security_groups)
    else:
        print('No additional security groups added')

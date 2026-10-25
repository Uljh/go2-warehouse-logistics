"""Restrict contact-trained locomotion transfer to the carried arm state."""


def carry_policy_eligible(*,original_z1,payload_held,phase,arm_state):
    return bool(original_z1 is True and payload_held is True and arm_state=='carried'
        and phase in ('leaving_receiving','transporting','waiting_approach_corridor',
            'waiting_west_spine','waiting_row_1','waiting_row_2','waiting_row_3',
            'waiting_delivery_transit'))


def validate_contact_manifest(manifest):
    payload=manifest.get('contact_payload') or {}
    if not (manifest.get('actual_training_completed') is True
            and manifest.get('arm_model')=='z1'
            and manifest.get('observation_dimension')==48
            and manifest.get('action_dimension')==12
            and payload.get('runtime_state_writes')==0
            and payload.get('payload_joints')==[]
            and payload.get('force_guard_n')==20.
            and manifest.get('evaluation_survived')):
        raise ValueError('Carry expert requires completed original-Z1 free-contact training and evaluation')

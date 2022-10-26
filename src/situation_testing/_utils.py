from typing import Dict


def get_prot_value(sensitive_val: Dict):
    if 'protected' in sensitive_val.keys():
        return sensitive_val['protected']
    else:
        raise ValueError("missing 'protected' key in dict value for the sensitive attribute(s)")

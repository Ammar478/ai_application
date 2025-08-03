from collections import defaultdict

_sessions = defaultdict(dict)

def get_session(session_id):
    return _sessions[session_id]

def update_session(session_id, key, value):
    _sessions[session_id][key]= value
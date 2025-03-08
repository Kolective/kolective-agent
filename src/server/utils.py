import os
import orjson
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
wallet_dir = os.path.join(current_dir, '../../data')
wallet_dir = os.path.normpath(wallet_dir)
file_path = os.path.join(wallet_dir, 'wallet.json')

def _update_risk_profile(risk_profile: str, user_address: str):    
    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())
        
    for entry in wallet_data:
        if entry["user_address"] == user_address:
            entry["risk_profile"] = risk_profile
            with open(file_path, 'wb') as file:
                file.write(orjson.dumps(wallet_data, option=orjson.OPT_INDENT_2))


def _get_user_risk(user_address: str):
    with open(file_path, 'rb') as file:
        wallet_data = orjson.loads(file.read())

    for entry in wallet_data:
        if entry["user_address"] == user_address:
            return entry["risk_profile"]
    return None

def _parse_data_kol(kol_username: str):
    response = requests.get("https://kolective-backend.vercel.app/api/kol")
    data = response.json()
    for kol in data['kols']:
        if kol['username'] == kol_username:
            return kol['id']
    
    
    